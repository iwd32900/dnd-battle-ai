#!/usr/bin/env python

from __future__ import annotations # allows forward declaration of types
import itertools
import re
import numpy as np
import torch
import ppo_clip
rnd = np.random.default_rng()

class Dice:
    def __init__(self, XdY: str):
        m = re.search(r'([1-9][0-9]*)?d([1-9][0-9]*)([+-][1-9][0-9]*)?', XdY)
        g = m.groups()
        self.num_dice = int(g[0] or 1)
        self.dice_type = int(g[1])
        self.bonus = int(g[2] or 0)
    def roll(self) -> int:
        rolls = rnd.integers(low=1, high=self.dice_type, endpoint=True, size=self.num_dice)
        # D&D rules: even if bonus is negative, total can't fall below 1
        result = max(1, rolls.sum() + self.bonus)
        #print(rolls, result)
        return result
    def roll_ad(self, advantage: bool, disadvantage: bool) -> int:
        if advantage and not disadvantage:
            return max(self.roll(), self.roll())
        elif disadvantage and not advantage:
            return min(self.roll(), self.roll())
        else:
            return self.roll()

def roll(XdY: str):
    return Dice(XdY).roll()

class Character:
    def __init__(self, name: str, team: int, hp: int, ac: int, actions: list[Action]):
        self.name = name
        self.team = team
        self.max_hp = self.hp = hp
        self.ac = ac
        self.actions = actions
        self.start_of_round() # initializes some properties

    def start_of_round(self):
        self.dodging = False

    def end_of_encounter(self, env):
        pass

    def damage(self, dmg_hp: int, dmg_type: str):
        "Apply damage to this character"
        self.hp = max(0, self.hp - dmg_hp)

    def heal(self, heal_hp: int):
        "Apply healing to this character"
        if self.hp <= 0:
            self.hp = 1
        else:
            self.hp = min(self.max_hp, self.hp + heal_hp)

class RandomCharacter(Character):
    def act(self, env):
        actions = [a for a in self.actions if not a.is_forbidden()]
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

MAX_TURNS = 100
MAX_CHARS = 3

class PPOStrategy:
    def __init__(self, n_acts):
        self.n_acts = n_acts * MAX_CHARS
        self.obs_dim = 5 * MAX_CHARS
        self.act_crit = ppo_clip.MLPActorCritic(self.obs_dim, self.n_acts, hidden_sizes=[32])
        self.buf = ppo_clip.PPOBuffer(self.obs_dim, self.n_acts, act_dim=None, size=1000 * 5 * MAX_TURNS)
        self.optim = ppo_clip.PPOAlgo(self.act_crit) #, pi_lr=1e-4)
        self.encounters = 0

    def end_of_encounter(self):
        self.encounters += 1
        # if self.encounters % 1000 == 0:
        #     self.optim.update(self.buf)

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

    def end_of_encounter(self, env):
        self.buf.finish_path(self.get_reward(env))
        self.ppo_strat.end_of_encounter()

    def get_obs(self, env):
        obs = []
        for c in env.characters:
            obs.extend([
                #c.team == self.team,    # on our team? same for whole training run, so useless
                c == self,              # ourself? maybe useful when 1 AI plays many monsters
                c.max_hp / self.max_hp, # stronger or weaker than us?
                c.hp / c.max_hp,        # hurt or healthy?
                (c.ac - 10) / 10,       # armor class
                c.dodging,              # taking Dodge action?
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

    def act(self, env):
        chars_acts = []
        fbn = []
        for c in env.characters:
            for a in self.actions:
                chars_acts.append((c,a))
                fbn.append(a.is_forbidden() or not a.plausible_target(self, c))
        fbn = np.array(fbn)
        if fbn.all():
            #print(f"{self.name} has no allowable action/target pairs")
            return

        obs = self.get_obs(env)
        act, val, logp, pi = self.act_crit.step(obs, fbn)
        rew = self.get_reward(env)
        self.buf.store(obs, fbn, act, rew, val, logp)
        act_idx = act.item()
        target, action = chars_acts[act_idx]
        action(actor=self, target=target)

class Action:
    def is_forbidden(self):
        return False

    def plausible_target(self, actor: Character, target: Character):
        return True

class Dodge(Action):
    name = 'Dodge'

    def __call__(self, actor: Character, target: Character):
        # target is irrelevant
        actor.dodging = True
        #print(f"{actor.name} used {self.name}")

class MeleeAttack(Action):
    def __init__(self, name: str, to_hit: int, dmg_dice: str, dmg_type: str):
        self.name = name
        self.to_hit = Dice(f'd20+{to_hit}')
        self.dmg_dice = Dice(dmg_dice)
        self.dmg_type = dmg_type

    def plausible_target(self, actor: Character, target: Character):
        return target.team != actor.team and target.hp > 0

    def __call__(self, actor: Character, target: Character):
        advntg = disadv = False
        if target.dodging:
            disadv = True
        attack_roll = self.to_hit.roll_ad(advntg, disadv)
        if attack_roll >= target.ac:
            dmg_roll = self.dmg_dice.roll()
            target.damage(dmg_roll, self.dmg_type)
            #print(f"{actor.name} attacked {target.name} with {self.name} for {dmg_roll}")
        else:
            pass #print(f"{actor.name} attacked {target.name} with {self.name} and missed")

class HealingPotion(Action):
    def __init__(self, name: str, heal_dice: str, uses: int = 1):
        self.name = name
        self.heal_dice = Dice(heal_dice)
        self.uses = uses

    def is_forbidden(self):
        return (self.uses <= 0)

    def plausible_target(self, actor: Character, target: Character):
        return target.team == actor.team and target.hp < target.max_hp

    def __call__(self, actor: Character, target: Character):
        if self.is_forbidden():
            return
        heal_roll = self.heal_dice.roll()
        target.heal(heal_roll)
        self.uses -= 1
        #print(f"{actor.name} used {self.name} on {target.name} for {heal_roll}")

class Environment:
    def __init__(self, characters):
        self.characters = characters

    def run(self):
        chars = list(self.characters)
        rnd.shuffle(chars)
        while True:
            #print("== top of round ==")
            #print({c.name: c.hp for c in chars})
            for actor in chars:
                actor.start_of_round()
                if actor.hp <= 0:
                    continue
                actor.act(self)
            active_teams = set(c.team for c in chars if c.hp > 0)
            if len(active_teams) <= 1:
                break
        #print({c.name: c.hp for c in chars})
        for actor in chars:
            actor.end_of_encounter(self)
        return (0 in active_teams)

def run_epoch(args):
    n, hero_strat, goblin_strat = args
    hero = lambda: PPOCharacter(hero_strat, name='Hero', team=0, hp=20, ac=15, actions=[
            Dodge(),
            MeleeAttack('long sword', +3, '1d8+1', 'slashing'),
            #MeleeAttack('greataxe', +3, '1d12+1', 'slashing'),
            #HealingPotion('potion of healing', '2d4+2', uses=3),
            HealingPotion('potion of greater healing', '4d4+4', uses=3),
        ])
    goblin = lambda i: RandomCharacter(f'Goblin {i}', team=1, hp=roll('2d6'), ac=15, actions=[
    #goblin = lambda i: PPOCharacter(goblin_strat, survival=0, name=f'Goblin {i}', team=1, hp=roll('2d6'), ac=15, actions=[
            Dodge(),
            MeleeAttack('scimitar', +4, '1d6+2', 'slashing')
        ])
    wins = 0
    for i in range(n):
        env = Environment([hero(), goblin(1), goblin(2)])
        win = env.run()
        if win: wins += 1
    return (wins, hero_strat.buf, goblin_strat.buf)

def merge_ppo_data(ppo_buffers):
    data = [x.get() for x in ppo_buffers]
    out = {}
    for key in data[0].keys():
        out[key] = torch.cat([x[key] for x in data])
    return out

from torch import multiprocessing
def main():
    epochs = 20
    ncpu = 4 # using 8 doesn't seem to help on an M1
    hero_strat = PPOStrategy(3)
    goblin_strat = PPOStrategy(2)
    with multiprocessing.Pool(ncpu) as pool:
        for epoch in range(epochs):
            results = pool.map(run_epoch, ((1000//ncpu, hero_strat, goblin_strat) for _ in range(ncpu)))
            wins = np.array([x[0] for x in results]) * ncpu
            print(f"{wins.mean():.0f} ± {wins.std():.1f}")
            hero_data = merge_ppo_data([x[1] for x in results])
            hero_strat.update(hero_data)
            hero_strat.buf.reset() # not sure if this is needed
            # goblin_data = merge_ppo_data([x[2] for x in results])
            # goblin_strat.update(goblin_data)
            # goblin_strat.buf.reset() # not sure if this is needed

if __name__ == '__main__':
    main()