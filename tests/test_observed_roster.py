"""Check observed-roster boundaries, unique species, and missing lookup handling."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pokemon_challenge.features.observed_roster import p2_observed_roster_stats,SPECIES_BASE_STATS

class ObservedRosterTests(unittest.TestCase):
    def test_unique_species_and_lead_included(self):
        battle={'p2_lead_details':{'name':'snorlax'},'battle_timeline':[
            {'turn':t,'p2_pokemon_state':{'name':name}}
            for t,name in [(1,'pikachu'),(2,'pikachu'),(30,'snorlax'),(31,'mew')]]}
        result=p2_observed_roster_stats(battle)
        for stat in ('hp','atk','def','spe'):
            self.assertEqual(result[f'p2_mean_{stat}'],(SPECIES_BASE_STATS['snorlax'][stat]+SPECIES_BASE_STATS['pikachu'][stat])/2)

    def test_only_first_30_entries(self):
        turns=[{'turn':1,'p2_pokemon_state':{'name':'pikachu'}} for _ in range(30)]
        turns.append({'turn':30,'p2_pokemon_state':{'name':'mew'}})
        self.assertEqual(p2_observed_roster_stats({'battle_timeline':turns})['p2_mean_hp'],SPECIES_BASE_STATS['pikachu']['hp'])

    def test_unknown_species_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError,'Missing base stats'):
            p2_observed_roster_stats({'p2_lead_details':{'name':'unknown_species'}})

    def test_empty_roster(self):
        self.assertEqual(p2_observed_roster_stats({}),dict.fromkeys(['p2_mean_hp','p2_mean_atk','p2_mean_def','p2_mean_spe'],0.0))
