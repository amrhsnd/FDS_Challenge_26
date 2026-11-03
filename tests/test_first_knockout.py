"""Check first-faint direction, timing, ties, and observation boundaries."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pokemon_challenge.features.first_knockout import first_ko_advantage

class FirstKnockoutTests(unittest.TestCase):
    @staticmethod
    def event(turn,side,**state):
        return {'turn':turn,f'{side}_pokemon_state':state}

    def test_direction_and_timing(self):
        for turn in (1,15,30):
            for side,sign in [('p1',-1),('p2',1)]:
                with self.subTest(turn=turn,side=side):
                    self.assertAlmostEqual(first_ko_advantage({'battle_timeline':[self.event(turn,side,hp_pct=0)]}),sign*(31-turn)/30)

    def test_status_faint_and_earliest_event(self):
        timeline=[self.event(20,'p2',status='fnt'),self.event(5,'p1',hp_pct=0),self.event(6,'p1',hp_pct=0)]
        self.assertAlmostEqual(first_ko_advantage({'battle_timeline':timeline}),-26/30)

    def test_tie_and_no_faint(self):
        for timeline in ([],[self.event(1,'p1',hp_pct=1,status='slp')],[self.event(10,'p1',hp_pct=0),self.event(10,'p2',status='fnt')]):
            self.assertEqual(first_ko_advantage({'battle_timeline':timeline}),0)

    def test_only_first_30_entries_and_valid_turn_numbers(self):
        self.assertEqual(first_ko_advantage({'battle_timeline':[self.event(31,'p1',hp_pct=0),self.event(0,'p2',hp_pct=0)]}),0)
        timeline=[self.event(1,'p1',hp_pct=1)]*30+[self.event(30,'p2',hp_pct=0)]
        self.assertEqual(first_ko_advantage({'battle_timeline':timeline}),0)
