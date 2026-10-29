"""Stateless numeric features from permitted battle observations."""
import numpy as np


def team_stats(battle):
    """Summarize all six base stats of P1's team and P2's initial lead."""
    team = battle.get("p1_team_details") or []
    lead = battle.get("p2_lead_details") or {}
    features = {"team_size": len(team)}
    for stat in ("hp", "atk", "def", "spa", "spd", "spe"):
        values = [p.get(f"base_{stat}", np.nan) for p in team]
        features[f"p1_mean_{stat}"] = float(np.mean(values)) if values else np.nan
        features[f"p2_lead_{stat}"] = lead.get(f"base_{stat}", np.nan)
    return features


def timeline_summary(battle):
    """Summarize recorded active states; HP is not total remaining team health."""
    turns = battle.get("battle_timeline") or []
    features = {"observed_turns": len(turns)}
    for side in ("p1", "p2"):
        states = [t.get(f"{side}_pokemon_state") or {} for t in turns]
        moves = [t.get(f"{side}_move_details") for t in turns]
        hp = [s["hp_pct"] for s in states if s.get("hp_pct") is not None]
        names = [s.get("name") for s in states]
        features[f"{side}_mean_active_hp"] = float(np.mean(hp)) if hp else np.nan
        features[f"{side}_last_active_hp"] = states[-1].get("hp_pct", np.nan) if states else np.nan
        features[f"{side}_distinct_observed_species"] = len({n for n in names if n is not None})
        features[f"{side}_active_name_changes"] = sum(a is not None and b is not None and a != b for a, b in zip(names, names[1:]))
        features[f"{side}_null_moves"] = sum(m is None for m in moves)
        features[f"{side}_status_observations"] = sum(s.get("status") not in (None, "nostatus", "") for s in states)
        for category in ("physical", "special", "status"):
            features[f"{side}_{category}_moves"] = sum((m or {}).get("category") == category for m in moves)
    return features
