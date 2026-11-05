"""Emily-derived features with fold-local species lookup and strict observation bounds."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

ROOT = Path(__file__).resolve().parents[2]
AIMILIA = ROOT / "notebooks/aimilia"
if str(AIMILIA) not in sys.path:
    sys.path.append(str(AIMILIA))
import features_static as static
import features_timeline as timeline
import features_effects as effects
import features_roster as roster
import features_flow as flow
import features_matchup as matchup
import features_speed as speed
import features_damage as damage
import features_hazards as hazards
import features_moves as moves
from pokedex import build_pokedex
from study import observed_battle


def clean_observation(battle):
    """Exclude labels, IDs and turns after 30; retain missingness rather than invent HP."""
    b = observed_battle(battle)
    numbers = [t['turn'] for t in b['battle_timeline']]
    if numbers != list(range(1, 31)):
        raise ValueError('Emily feature builders require exactly turns 1 through 30.')
    b['battle_id'] = 0  # Temporary join key, never a predictor.
    for t in b['battle_timeline']:
        for side in ('p1', 'p2'):
            hp = t[f'{side}_pokemon_state'].get('hp_pct')
            if hp is None or not np.isfinite(hp) or not 0 <= hp <= 1:
                raise ValueError('Expected finite active HP fractions in [0, 1].')
    return b


class EmilyFeatures(TransformerMixin, BaseEstimator):
    """Fit species metadata on training battles only; transform each battle independently."""
    def __init__(self, chart_path, bundle='all'):
        self.chart_path = chart_path
        self.bundle = bundle

    def fit(self, X, y=None):
        if self.bundle not in ('baseline', 'keep3', 'all'):
            raise ValueError('Unknown feature bundle.')
        # Labels and all non-permitted keys are removed even during fit.
        self.dex_ = build_pokedex([clean_observation(b) for b in X['battle']])
        self.chart_ = matchup.read_chart(self.chart_path)
        self.feature_names_in_ = np.asarray(['battle'], dtype=object)
        self.n_features_in_ = 1
        sample = self._matrix(X.iloc[:1])
        self.columns_ = list(sample.columns)
        return self

    def _matrix(self, X):
        rows = []
        for raw in X['battle']:
            b = clean_observation(raw)
            row = {}
            def add(group, values, predicate=lambda key: True):
                row.update({f'{group}__{k}': v for k, v in values.items()
                            if k not in ('battle_id', 'player_won') and predicate(k)})
            add('static', static.battle_to_row(b, labelled=False))
            add('timeline', timeline.battle_to_row(b))
            add('roster', roster.battle_to_row(b))
            effect = effects.battle_to_row(b)
            add('boost', effect, lambda k: 'boost' in k)
            if self.bundle in ('keep3', 'all'):
                add('matchup', matchup.battle_to_row(b, self.chart_, self.dex_))
                add('damage', damage.battle_to_row(b, self.chart_, self.dex_))
                add('hazards', hazards.battle_to_row(b))
            if self.bundle == 'all':
                add('effect_rates', effect, lambda k: 'boost' not in k)
                add('flow', flow.battle_to_row(b))
                add('speed', speed.battle_to_row(b, self.dex_))
                add('moves', moves.battle_to_row(b))
                # Coverage makes missing species resolution explicit to the model.
                for side in ('p1', 'p2'):
                    local = {p['name'] for p in b['p1_team_details']} | {b['p2_lead_details']['name']}
                    names = {t[f'{side}_pokemon_state']['name'] for t in b['battle_timeline']}
                    row[f'coverage__{side}_unknown_species'] = len(names - local - set(self.dex_))
            rows.append(row)
        result = pd.DataFrame(rows, index=X.index, dtype=float)
        if np.isinf(result.to_numpy()).any():
            raise ValueError('Infinite feature value.')
        return result

    def transform(self, X):
        check_is_fitted(self, 'columns_')
        result = self._matrix(X)
        if list(result.columns) != self.columns_:
            raise ValueError('Feature schema changed.')
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'columns_')
        return np.asarray(self.columns_, dtype=object)
