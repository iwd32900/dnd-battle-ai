#!/usr/bin/env python

from __future__ import annotations # allows forward declaration of types
import csv
import enum
import os
import re
import time
import numpy as np
import torch
from torch import multiprocessing
import ppo_clip
rnd = np.random.default_rng()

MAX_TURNS = 100
MAX_CHARS = 5
OBS_SIZE = 4 * MAX_CHARS
HP_SCALE = 20 # roughly, max HP across all entities in the battle (but a fixed constant, not rolled dice!)

Ability = enum.IntEnum('Ability', 'STR DEX CON INT WIS CHA', start=0)

epoch_id = encounter_id = round_id = -1
actions_csv = csv.writer(open(f"actions_{os.getpid()}.csv", "w"))
actions_csv.writerow('epoch encounter round actor action target t_dodging t_weakest raw_hp obs_hp'.split())
outcomes_csv = csv.writer(open(f"outcomes_{os.getpid()}.csv", "w"))
outcomes_csv.writerow('epoch encounter num_rounds actor team team_win max_hp final_hp'.split())

class Dice:
    def __init__(self, XdY: str):
        m = re.search(r'([1-9][0-9]*)?d([1-9][0-9]*)([+-][1-9][0-9]*)?', XdY)
        g = m.groups()
        self.num_dice = int(g[0] or 1)
        self.dice_type = int(g[1])
        self.bonus = int(g[2] or 0)
    def roll(self, crit_hit: bool = False) -> int:
        rolls = rnd.integers(low=1, high=self.dice_type, endpoint=True, size=self.num_dice * (2 if crit_hit else 1))
        # D&D rules: even if bonus is negative, total can't fall below 1
        result = max(1, rolls.sum() + self.bonus)
        #print(rolls, result)
        return result

class D20:
    def __init__(self, bonus: int):
        self.bonus = bonus
    def roll(self, advantage: bool = False, disadvantage: bool = False):
        rolls = rnd.integers(low=1, high=20, endpoint=True, size=2)
        if advantage and not disadvantage:
            base_roll = rolls.max()
        elif disadvantage and not advantage:
            base_roll = rolls.min()
        else:
            base_roll = rolls[0]
        mod_roll = max(1, base_roll + self.bonus)
        return mod_roll, base_roll

def roll(XdY: str):
    return Dice(XdY).roll()

class Character:
    def __init__(self, name: str, team: int, hp: int, ac: int, actions: list[Action],
                 ability_mods: list[int] = [0]*6, saving_throws: list[int] = None,
                 spells: list[int] = [0]*9, spell_attack=2, spell_save=10):
        self.name = name
        self.team = team
        self.max_hp = self.hp = hp
        self.ac = ac
        self.actions = actions
        self.ability_mods = list(ability_mods)
        self.saving_throws = list(saving_throws or ability_mods) # if None, default == to ability_mods
        self.max_spells = np.array(spells, dtype=int)
        self.curr_spells = self.max_spells.copy()
        self.spell_attack = D20(spell_attack)
        self.spell_save_dc = spell_save
        self.unconscious = False
        self.start_of_round() # initializes some properties

    def start_of_round(self):
        self.dodging = False

    def end_of_encounter(self, env):
        pass

    def saving_throw(self, save_dc: int, ability: Ability, advantage: bool = False, disadvantage: bool = False):
        "Returns True if the save succeeds and False if it fails"
        if self.unconscious and ability in [Ability.STR, Ability.DEX]:
            return False
        mod_roll, nat_roll = D20(self.saving_throws[ability]).roll(advantage=advantage, disadvantage=disadvantage)
        if nat_roll == 1: return False
        elif nat_roll == 20: return True
        else: return mod_roll >= save_dc

    def damage(self, dmg_hp: int, dmg_type: str):
        "Apply damage to this character"
        self.hp = max(0, self.hp - dmg_hp)
        self.unconscious = False

    def heal(self, heal_hp: int):
        "Apply healing to this character"
        if self.hp <= 0:
            self.hp = 1
        else:
            self.hp = min(self.max_hp, self.hp + heal_hp)

class RandomCharacter(Character):
    def act(self, env):
        actions = [a for a in self.actions if not a.is_forbidden(self, env)]
        if not actions:
            #print(f"{self.name} has no allowable actions")
            return
        action = rnd.choice(actions)
        targets = [c for c in env.characters if action.plausible_target(self, c)]
        if not targets:
            #print(f"{self.name} could not find a target for {action.name}")
            return
        target = rnd.choice(targets)
        action(actor=self, target=target)

class PPOStrategy:
    def __init__(self, n_acts):
        self.n_acts = n_acts * MAX_CHARS
        self.obs_dim = OBS_SIZE
        self.act_crit = ppo_clip.MLPActorCritic(self.obs_dim, self.n_acts, hidden_sizes=[32])
        self.act_crit.share_memory() # docs say this is required, but doesn't seem to be?
        # https://bair.berkeley.edu/blog/2021/07/14/mappo/ suggests that smaller clip (0.2) and
        # fewer iters (5-15) stabilizes learning with PPO in multi-agent settings?
        # So far, I don't see a benefit.
        self.optim = ppo_clip.PPOAlgo(self.act_crit) # pi_lr=3e-4, vf_lr=1e-3, clip_ratio=0.2, train_pi_iters=15, train_v_iters=15
        self.encounters = 0

    def alloc_buf(self):
        # Wait to allocate the buffers until we're in worker processes, so we don't trample the same memory
        self.buf = ppo_clip.PPOBuffer(self.obs_dim, self.n_acts, act_dim=None, size=1000 * OBS_SIZE * MAX_TURNS)

    def end_of_encounter(self):
        self.encounters += 1

    def update(self, data):
        self.optim.update(data)

class PPOCharacter(Character):
    def __init__(self, ppo_strat, survival=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert ppo_strat.n_acts == len(self.actions) * MAX_CHARS
        self.ppo_strat = ppo_strat
        self.act_crit = ppo_strat.act_crit
        self.buf = ppo_strat.buf
        self.survival = survival
        self.old_hp_score = 0
        self.prev_OFAVL = None # state tuple from previous reward

    def end_of_encounter(self, env):
        # all consequences of last action are now apparent, so can calc reward for it
        self.save_experience(env)
        self.buf.finish_path(self.get_reward(env))
        self.ppo_strat.end_of_encounter()

    def get_obs(self, env):
        obs = []
        for c in env.characters:
            obs.extend([
                #c.team == self.team,           # On our team? Same for whole training run, so useless.
                #(c.ac - 10) / 10,              # Armor class.  Right now, does not change.
                c == self,                      # Ourself? maybe useful when 1 AI plays many monsters
                (c.max_hp - c.hp) / HP_SCALE,   # Absolute hp lost -- we can track this as a player.
                c.unconscious,                  # Unconscious (not counting hp <= 0)
                c.dodging,                      # Taking Dodge action?
                # Below this point is cheating -- info not available to players, only DM
                #c.hp / HP_SCALE,               # current absolute health
                #c.max_hp / self.max_hp,        # Stronger or weaker than us? Varies if hp are rolled.
            ])
        return torch.tensor(obs, dtype=torch.float32)

    def get_hp_reward(self, env):
        """
        Reward characters when the enemies lose hp, or their team gains hp.
        Enemies losing 100% of their hp is worth +1,
        team losing 100% of their hp is worth -1.
        Each teammate death is worth an additional -1/team_size.
        At the start of the battle, with no dead and no hp losses, score should be zero.
        """
        chars = env.characters
        hp = np.empty(len(chars), dtype=float)
        max_hp = np.empty(len(chars), dtype=float)
        team = np.empty(len(chars), dtype=bool)
        for i, char in enumerate(chars):
            hp[i] = char.hp
            max_hp[i] = char.max_hp
            team[i] = (char.team == self.team)
        # All hp losses are equal, whether from a weak character or a strong one
        team_frac = hp[team].sum() / max_hp[team].sum()
        opp_frac = hp[~team].sum() / max_hp[~team].sum()
        team_size = team.sum()
        team_deaths = (hp[team] == 0).sum()
        # If `survival` is 0, there's no special attempt to avoid deaths.
        # If `survival` is 1, it's better to have 2 characters at 1 hp than one dead and one full.
        # The default of 0.5 means avoiding death is worth half of a teammate's hp.
        # Positive team is ahead, negative team is behind, zero is balanced loss of hp
        hp_score = team_frac - opp_frac - self.survival*team_deaths/team_size
        # Positive team has gained ground, negative team has lost ground, zero is no or balanced changes
        hp_delta = hp_score - self.old_hp_score
        self.old_hp_score = hp_score
        return hp_delta

    def get_reward(self, env):
        return self.get_hp_reward(env)

    def save_experience(self, env):
        rew = self.get_reward(env)
        if self.prev_OFAVL is None:
            # First action.  Any HP losses before this are independent of our actions,
            # so shouldn't count toward our rewards.
            pass
        else:
            obs, fbn, act, val, logp = self.prev_OFAVL
            self.buf.store(obs, fbn, act, rew, val, logp)
        self.prev_OFAVL = None

    def act(self, env):
        # all consequences of last action are now apparent, so can calc reward for it
        self.save_experience(env)

        chars_acts = []
        fbn = []
        for c in env.characters:
            for a in self.actions:
                chars_acts.append((c,a))
                fbn.append(a.is_forbidden(self, env) or not a.plausible_target(self, c))
        fbn = np.array(fbn)
        if fbn.all():
            #print(f"{self.name} has no allowable action/target pairs")
            return

        with torch.no_grad():
            obs = self.get_obs(env)
            act, val, logp, pi = self.act_crit.step(obs, fbn)
            act_idx = act.item()
            target, action = chars_acts[act_idx]
        self.prev_OFAVL = (obs, fbn, act, val, logp)
        action(actor=self, target=target, env=env)

class Action:
    def is_forbidden(self, actor: Character, env: Environment):
        return False

    def plausible_target(self, actor: Character, target: Character):
        return True

    def _self_only(self, actor: Character, target: Character):
        # This is a good choice for Actions without an explicit target, like Dodge, so there's just one unique choice
        return actor == target

    def _conscious_ally(self, actor: Character, target: Character):
        return target.team == actor.team and target.hp > 0

    def _unconscious_ally(self, actor: Character, target: Character):
        return target.team == actor.team and target.unconscious

    def _conscious_enemy(self, actor: Character, target: Character):
        return target.team != actor.team and target.hp > 0

class Dodge(Action):
    name = 'Dodge'
    plausible_target = Action._self_only
    # This way, there's one unique (legal) Dodge action, instead of one per opponent

    def __call__(self, actor: Character, target: Character, env: Environment):
        # target is irrelevant
        actor.dodging = True
        #print(f"{actor.name} used {self.name}")
        actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, self.name, target.name, False, False, 0, 0])

class Awaken(Action):
    name = 'Awaken'
    plausible_target = Action._unconscious_ally

    def __call__(self, actor: Character, target: Character, env: Environment):
        target.unconscious = False
        actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, self.name, target.name, False, False, 0, 0])

class MeleeAttack(Action):
    def __init__(self, name: str, to_hit: int, dmg_dice: str, dmg_type: str):
        self.name = name
        self.to_hit = D20(to_hit)
        self.dmg_dice = Dice(dmg_dice)
        self.dmg_type = dmg_type

    def plausible_target(self, actor: Character, target: Character):
        return target.team != actor.team and target.hp > 0

    def __call__(self, actor: Character, target: Character, env: Environment):
        advntg = disadv = False
        if target.dodging:
            disadv = True
        t_weakest = all(target.hp <= other.hp for other in env.characters if self.plausible_target(actor, other))
        # all([]) == True, which seems ok
        attack_roll, nat_roll = self.to_hit.roll(advntg, disadv)
        if (attack_roll >= target.ac or nat_roll == 20) and nat_roll != 1:
            crit_hit = (nat_roll == 20) or target.unconscious
            dmg_roll = self.dmg_dice.roll(crit_hit=crit_hit)
            before_hp = target.hp
            target.damage(dmg_roll, self.dmg_type)
            after_hp = target.hp
            #print(f"{actor.name} attacked {target.name} with {self.name} for {dmg_roll}")
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, self.name, target.name, target.dodging, t_weakest, -dmg_roll, after_hp - before_hp])
        else:
            #print(f"{actor.name} attacked {target.name} with {self.name} and missed")
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, self.name, target.name, target.dodging, t_weakest, 0, 0])

class HealingPotion(Action):
    def __init__(self, name: str, heal_dice: str, uses: int = 1):
        self.name = name
        self.heal_dice = Dice(heal_dice)
        self.uses = uses

    def is_forbidden(self, actor: Character, env: Environment):
        return (self.uses <= 0)

    def plausible_target(self, actor: Character, target: Character):
        return target.team == actor.team and target.hp < target.max_hp

    def __call__(self, actor: Character, target: Character, env: Environment):
        if self.is_forbidden(actor, env):
            return
        t_weakest = all(target.hp <= other.hp for other in env.characters if self.plausible_target(actor, other))
        # all([]) == True, which seems ok
        heal_roll = self.heal_dice.roll()
        before_hp = target.hp
        target.heal(heal_roll)
        after_hp = target.hp
        self.uses -= 1
        #print(f"{actor.name} used {self.name} on {target.name} for {heal_roll}")
        actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, self.name, target.name, target.dodging, t_weakest, heal_roll, after_hp - before_hp])

class Spell(Action):
    # def __init__(self, name: str, level: int, concentration: bool = False):
    #     self.name = name
    #     self.level = level # 0 for cantrips
    #     self.concentration = concentration

    def is_forbidden(self, actor: Character, env: Environment):
        return self.level > 0 and actor.curr_spells[self.level-1] <= 0

    def _consume_slot(self, actor: Character):
        if self.level > 0: actor.curr_spells[self.level-1] -= 1

    def _spell_attack(self, actor: Character, target: Character, env: Environment, dmg_dice: Dice, dmg_type: str):
        advntg = disadv = False
        if target.dodging:
            disadv = True # TODO: does this affect spells?
        t_weakest = all(target.hp <= other.hp for other in env.characters if self.plausible_target(actor, other))
        # all([]) == True, which seems ok
        attack_roll, nat_roll = actor.spell_attack.roll(advntg, disadv)
        if (attack_roll >= target.ac or nat_roll == 20) and nat_roll != 1:
            crit_hit = (nat_roll == 20) or target.unconscious # critical hits work with spell attacks too
            dmg_roll = dmg_dice.roll(crit_hit=crit_hit)
            before_hp = target.hp
            target.damage(dmg_roll, dmg_type)
            after_hp = target.hp
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, self.name, target.name, target.dodging, t_weakest, -dmg_roll, after_hp - before_hp])
        else:
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, self.name, target.name, target.dodging, t_weakest, 0, 0])

    def __call__(self, actor: Character, target: Character, env: Environment):
        if self.is_forbidden(actor, env): return
        self._consume_slot(actor)
        self.call(actor, target, env)

class MageArmor(Spell):
    name = 'Mage Armor'
    level = 1
    concentration = False
    plausible_target = Action._conscious_ally

    def call(self, actor: Character, target: Character, env: Environment):
        target.ac = 13 + target.ability_mods[Ability.DEX]
        actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, self.name, target.name, target.dodging, False, 0, 0])

class MagicMissle(Spell):
    name = 'Magic Missle'
    level = 1
    concentration = False
    plausible_target = Action._conscious_enemy

    def call(self, actor: Character, target: Character, env: Environment):
        # Automatically hits.  TODO:  should be able to target multiple opponents
        t_weakest = all(target.hp <= other.hp for other in env.characters if self.plausible_target(actor, other))
        dmg_roll = roll('3d4+3')
        before_hp = target.hp
        target.damage(dmg_roll, 'force')
        after_hp = target.hp
        actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, self.name, target.name, target.dodging, t_weakest, -dmg_roll, after_hp - before_hp])

class RayOfFrost(Spell):
    name = 'Ray of Frost'
    level = 0 # cantrip
    concentration = False
    plausible_target = Action._conscious_enemy

    def call(self, actor: Character, target: Character, env: Environment):
        self._spell_attack(actor, target, env, Dice('1d8'), 'cold')

class Sleep(Spell):
    name = 'Sleep'
    level = 1
    concentration = False
    plausible_target = Action._self_only
    # maybe this should be _conscious_enemy instead, if we factor out actions vs. targets?

    def call(self, actor: Character, target: Character, env: Environment):
        orig_hp = hp = roll('5d8')
        targets = [c for c in env.characters if c.team != actor.team and c.hp > 0 and not c.unconscious]
        targets.sort(key=lambda c: c.hp)
        t_names = []
        for target in targets:
            if target.hp <= hp:
                target.unconscious = True
                hp -= target.hp
                t_names.append(target.name)
            else:
                break
        actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, self.name, '/'.join(t_names), False, False, -orig_hp, -(orig_hp - hp)])

class Environment:
    def __init__(self, characters):
        assert len(characters) == MAX_CHARS
        self.characters = characters

    def run(self):
        chars = list(self.characters)
        # Initiative:
        rnd.shuffle(chars) # this ensures ties are broken randomly
        chars.sort(key=lambda c: D20(c.ability_mods[Ability.DEX]).roll(), reverse=True) # key() is called once per item and cached
        global round_id
        round_id = 0
        while True:
            #print("== top of round ==")
            #print({c.name: c.hp for c in chars})
            for actor in chars:
                actor.start_of_round()
                if actor.hp <= 0 or actor.unconscious:
                    continue
                actor.act(self)
            active_teams = set(c.team for c in chars if c.hp > 0)
            if len(active_teams) <= 1:
                break
            round_id += 1
        #print({c.name: c.hp for c in chars})
        for actor in chars:
            actor.end_of_encounter(self)
            outcomes_csv.writerow([
                epoch_id, encounter_id, round_id+1,
                actor.name, actor.team, (actor.team in active_teams),
                actor.max_hp, actor.hp
            ])
        return (0 in active_teams)

def init_workers(strats):
    #print(rnd.random()) # confirm that each process has unique random seed
    global strategies
    strategies = strats
    for s in strategies:
        s.alloc_buf()

def run_epoch(args):
    epoch_id_, n = args
    global strategies, epoch_id, encounter_id
    epoch_id = epoch_id_
    for s in strategies:
        s.buf.reset()
    fighter_lvl2 = lambda i: PPOCharacter(strategies[0], name=f'Fighter {i}', team=0, hp=20, ac=18, actions=[
            Dodge(),
            Awaken(),
            MeleeAttack('long sword', +5, '1d8+3', 'slashing'),
            HealingPotion('potion of healing', '2d4+2', uses=3),
            #HealingPotion('potion of greater healing', '4d4+4', uses=3),
        ],
        ability_mods=[3,2,2,-1,1,0], saving_throws=[5,2,4,-1,1,0])
    wizard_lvl2 = lambda i: PPOCharacter(strategies[1], name=f'Wizard {i}', team=0, hp=14, ac=12, actions=[
            Dodge(),
            Awaken(),
            #MeleeAttack('quarterstaff (two-handed)', +1, '1d8-1', 'bludgeoning'), # worse hit & damage than dagger
            MeleeAttack('dagger', +4, '1d4+2', 'piercing'),
            MageArmor(),
            MagicMissle(),
            RayOfFrost(), # better hit & same damage as dagger
            Sleep(),
            # Witch Bolt -- too much state tracking!
            # Charm Person?
        ],
        ability_mods=[-1,2,2,3,1,0], saving_throws=[-1,2,2,5,3,0],
        spells=[3,0,0,0,0,0,0,0,0], spell_attack=5, spell_save=13)
    #goblin = lambda i: RandomCharacter(f'Goblin {i}', team=1, hp=roll('2d6'), ac=15, actions=[
    goblin = lambda i: PPOCharacter(strategies[2], survival=0, name=f'Goblin {i}', team=1, hp=roll('2d6'), ac=15, actions=[
            Dodge(),
            Awaken(),
            MeleeAttack('scimitar', +4, '1d6+2', 'slashing')
        ],
        ability_mods=[-1,2,0,0,-1,1])
    wins = 0
    for encounter_id in range(n):
        env = Environment([fighter_lvl2(1), wizard_lvl2(1), goblin(1), goblin(2), goblin(3)])
        if env.run(): wins += 1
    return [s.buf for s in strategies] + [wins]

def run_update(args):
    ppo_buffers, strategy = args
    data = merge_ppo_data(ppo_buffers)
    strategy.update(data)

def merge_ppo_data(ppo_buffers):
    data = [x.get() for x in ppo_buffers]
    out = {}
    for key in data[0].keys():
        out[key] = torch.cat([x[key] for x in data])
    return out

def main():
    epochs = 100
    ncpu = 4 # using 8 doesn't seem to help on an M1
    strategies = [PPOStrategy(4), PPOStrategy(7), PPOStrategy(3)]
    with multiprocessing.Pool(ncpu, init_workers, (strategies,)) as pool:
        for epoch in range(epochs):
            t1 = time.time()
            results = pool.map(run_epoch, [(epoch, 1000//ncpu) for _ in range(ncpu)])
            # transpose results matrix so entries of same type are together
            results = list(zip(*results))
            t2 = time.time()
            wins = np.array(results[-1]) * ncpu
            pool.map(run_update, [(results[i], s) for i, s in enumerate(strategies)])
            t3 = time.time()
            print(f"Epoch {epoch:04d}:  {wins.mean():.0f} ± {wins.std():.0f} wins in {t2-t1:.1f} + {t3-t2:.1f} sec")

if __name__ == '__main__':
    main()