"""Names and definitions for numeric battle measurements.

P1 is the player being predicted. A p1_minus_p2 measurement is signed P1 minus
P2 unless the name explicitly gives the reverse direction. HP fraction is one
full member's health; team HP in bars sums those fractions over members. A
presumed roster treats unrevealed slots as healthy; an observed roster does not.
Rates are fractions, not percentages, unless a name explicitly says percent.

The private extraction-key mapping supports deterministic adapters. Public
names and documentation describe measurements, windows and assumptions only.
"""
from __future__ import annotations

import re

_RAW_FEATURES = {
    'selected': """
        alive_diff_at_turn30 first_ko_advantage p1_alive_at_turn30
        p1_boost_sum_at_turn30 p1_damage_surplus p1_dmg
        p1_heal p1_hp_weighted_base_special_attack p1_leechseed_rate
        p1_mean_atk p1_mean_boosts p1_mean_def
        p1_mean_hp p1_mean_spe p1_no_damage_moves
        p1_remaining_special_bulk p1_special_moves p1_switch
        p1_team_hp_pct_at_turn30 p2_boost_sum_at_turn30 p2_damage_surplus
        p2_dmg p2_heal p2_healthy_living_members
        p2_mean_atk p2_mean_boosts p2_mean_def
        p2_mean_hp p2_mean_spe p2_no_damage_moves
        p2_no_of_moves p2_team_hp_pct_at_turn30
    """.split(),
    'expanded_core': """
        p1_fainting_turns p1_max_observed_move_damage p1_mean_spa
        p1_mean_spd p1_negative_boosts_at_turn30 p1_no_of_moves
        p1_positive_boosts_at_turn30 p1_team_diversity p2_alive_at_turn30
        p2_fainting_turns p2_lead_atk p2_lead_def
        p2_lead_hp p2_lead_spa p2_lead_spd
        p2_lead_spe p2_max_observed_move_damage p2_negative_boosts_at_turn30
        p2_positive_boosts_at_turn30 p2_special_moves p2_switch
    """.split(),
    'roster_shape': """
        p1_base_atk_max p1_base_atk_min p1_base_atk_std
        p1_base_def_max p1_base_def_min p1_base_def_std
        p1_base_hp_max p1_base_hp_min p1_base_hp_std
        p1_base_spa_max p1_base_spa_min p1_base_spa_std
        p1_base_spd_max p1_base_spd_min p1_base_spd_std
        p1_base_spe_max p1_base_spe_min p1_base_spe_std
        p1_base_speed_ge100_count p1_berry_fraction_known_items p1_initial_physical_bulk_mean
        p1_initial_physical_bulk_std p1_initial_special_bulk_mean p1_initial_special_bulk_std
        p1_item_known_count p1_leftovers_fraction_known_items p1_level_mean
        p1_level_std p1_mean_minus_p2_lead_atk p1_mean_minus_p2_lead_def
        p1_mean_minus_p2_lead_hp p1_mean_minus_p2_lead_spa p1_mean_minus_p2_lead_spd
        p1_mean_minus_p2_lead_spe p1_special_to_physical_attack_mean p1_supplied_members
        p2_supplied_lead_level
    """.split(),
    'status_resources': """
        gap_retained_healthy_hp gap_retained_incapacitated_hp gap_retained_poison_hp
        gap_retained_status_hp p1_active_brn_rate p1_active_frz_rate
        p1_active_par_rate p1_active_psn_rate p1_active_slp_rate
        p1_active_tox_rate p1_active_unknown_status_rate p1_observation_age_max
        p1_observation_age_mean p1_retained_brn_count p1_retained_frz_count
        p1_retained_healthy_hp p1_retained_incapacitated_hp p1_retained_par_count
        p1_retained_poison_hp p1_retained_psn_count p1_retained_slp_count
        p1_retained_status_hp p1_retained_tox_count p1_retained_unknown_hp_count
        p1_retained_unknown_status_count p1_revealed_members p1_supplied_unobserved_members
        p2_active_brn_rate p2_active_frz_rate p2_active_par_rate
        p2_active_psn_rate p2_active_slp_rate p2_active_tox_rate
        p2_active_unknown_status_rate p2_observation_age_max p2_observation_age_mean
        p2_retained_brn_count p2_retained_frz_count p2_retained_healthy_hp
        p2_retained_incapacitated_hp p2_retained_par_count p2_retained_poison_hp
        p2_retained_psn_count p2_retained_slp_count p2_retained_status_hp
        p2_retained_tox_count p2_retained_unknown_hp_count p2_retained_unknown_status_count
        p2_revealed_members p2_unrevealed_slots
    """.split(),
    'recent_flow': """
        gap_t11_20_hp_gain gap_t11_20_hp_loss gap_t11_20_new_faints
        gap_t11_20_switches gap_t1_10_hp_gain gap_t1_10_hp_loss
        gap_t1_10_new_faints gap_t1_10_switches gap_t21_30_hp_gain
        gap_t21_30_hp_loss gap_t21_30_new_faints gap_t21_30_switches
        p1_hp_gain_late_minus_early p1_hp_loss_late_minus_early p1_t11_20_damaging_moves_rate
        p1_t11_20_hp_gain p1_t11_20_hp_loss p1_t11_20_new_faints
        p1_t11_20_no_move_rate p1_t11_20_priority_moves_rate p1_t11_20_rows
        p1_t11_20_special_moves_rate p1_t11_20_switches p1_t1_10_damaging_moves_rate
        p1_t1_10_hp_gain p1_t1_10_hp_loss p1_t1_10_new_faints
        p1_t1_10_no_move_rate p1_t1_10_priority_moves_rate p1_t1_10_rows
        p1_t1_10_special_moves_rate p1_t1_10_switches p1_t21_30_damaging_moves_rate
        p1_t21_30_hp_gain p1_t21_30_hp_loss p1_t21_30_new_faints
        p1_t21_30_no_move_rate p1_t21_30_priority_moves_rate p1_t21_30_rows
        p1_t21_30_special_moves_rate p1_t21_30_switches p2_hp_gain_late_minus_early
        p2_hp_loss_late_minus_early p2_t11_20_damaging_moves_rate p2_t11_20_hp_gain
        p2_t11_20_hp_loss p2_t11_20_new_faints p2_t11_20_no_move_rate
        p2_t11_20_priority_moves_rate p2_t11_20_rows p2_t11_20_special_moves_rate
        p2_t11_20_switches p2_t1_10_damaging_moves_rate p2_t1_10_hp_gain
        p2_t1_10_hp_loss p2_t1_10_new_faints p2_t1_10_no_move_rate
        p2_t1_10_priority_moves_rate p2_t1_10_rows p2_t1_10_special_moves_rate
        p2_t1_10_switches p2_t21_30_damaging_moves_rate p2_t21_30_hp_gain
        p2_t21_30_hp_loss p2_t21_30_new_faints p2_t21_30_no_move_rate
        p2_t21_30_priority_moves_rate p2_t21_30_rows p2_t21_30_special_moves_rate
        p2_t21_30_switches
    """.split(),
    'active_position': """
        final_observed_turn gap_active_atk_hp_stage gap_active_hp
        gap_active_hp_weighted_physical_attack_proxy gap_active_hp_weighted_special_attack_proxy gap_active_spa_hp_stage
        gap_active_spe_hp_stage gap_active_speed_proxy observed_turn_count
        p1_active_atk_hp_stage p1_active_atk_stage p1_active_def_hp_stage
        p1_active_def_stage p1_active_hp p1_active_hp_weighted_physical_attack_proxy
        p1_active_hp_weighted_special_attack_proxy p1_active_physical_attack_proxy p1_active_spa_hp_stage
        p1_active_spa_stage p1_active_spd_hp_stage p1_active_spd_stage
        p1_active_spe_hp_stage p1_active_spe_stage p1_active_special_attack_proxy
        p1_active_speed_proxy p1_active_stats_supplied p2_active_atk_hp_stage
        p2_active_atk_stage p2_active_def_hp_stage p2_active_def_stage
        p2_active_hp p2_active_hp_weighted_physical_attack_proxy p2_active_hp_weighted_special_attack_proxy
        p2_active_physical_attack_proxy p2_active_spa_hp_stage p2_active_spa_stage
        p2_active_spd_hp_stage p2_active_spd_stage p2_active_spe_hp_stage
        p2_active_spe_stage p2_active_special_attack_proxy p2_active_speed_proxy
        p2_active_stats_supplied
    """.split(),
    'aimilia_static': """
        diff_atk_max diff_atk_mean diff_def_max
        diff_def_mean diff_hp_max diff_hp_mean
        diff_spa_max diff_spa_mean diff_spd_max
        diff_spd_mean diff_spe_max diff_spe_mean
        p1_atk_max p1_atk_mean p1_def_max
        p1_def_mean p1_hp_max p1_hp_mean
        p1_level_mean p1_level_min p1_n_dual_type
        p1_n_special_item p1_n_types p1_spa_max
        p1_spa_mean p1_spd_max p1_spd_mean
        p1_spe_max p1_spe_mean p2_lead_atk
        p2_lead_def p2_lead_hp p2_lead_level
        p2_lead_n_types p2_lead_spa p2_lead_spd
        p2_lead_spe
    """.split(),
    'aimilia_timeline': """
        diff_hppct_final diff_hppct_mean diff_no_move_rate
        diff_status_rate diff_switches p1_faints
        p1_hppct_early_mean p1_hppct_final p1_hppct_late_mean
        p1_hppct_mean p1_hppct_min p1_hppct_trend
        p1_mean_accuracy p1_mean_base_power p1_n_species_seen
        p1_no_move_rate p1_slp_rate p1_status_move_rate
        p1_status_rate p1_switches p2_faints
        p2_hppct_early_mean p2_hppct_final p2_hppct_late_mean
        p2_hppct_mean p2_hppct_min p2_hppct_trend
        p2_mean_accuracy p2_mean_base_power p2_n_species_seen
        p2_no_move_rate p2_slp_rate p2_status_move_rate
        p2_status_rate p2_switches
    """.split(),
    'aimilia_effects': """
        diff_boost_mean diff_eff_substitute diff_eff_trapped
        p1_boost_defensive_mean p1_boost_final p1_boost_max
        p1_boost_mean p1_boost_offensive_mean p1_boost_speed_mean
        p1_eff_any p1_eff_attract p1_eff_clamp
        p1_eff_confusion p1_eff_destinybond p1_eff_encore
        p1_eff_leechseed p1_eff_perish p1_eff_protect
        p1_eff_substitute p1_eff_trapped p2_boost_defensive_mean
        p2_boost_final p2_boost_max p2_boost_mean
        p2_boost_offensive_mean p2_boost_speed_mean p2_eff_any
        p2_eff_attract p2_eff_clamp p2_eff_confusion
        p2_eff_destinybond p2_eff_encore p2_eff_leechseed
        p2_eff_perish p2_eff_protect p2_eff_substitute
        p2_eff_trapped
    """.split(),
    'aimilia_roster': """
        alive_diff_t10 alive_diff_t20 alive_diff_t30
        p1_alive_t10 p1_alive_t20 p1_alive_t30
        p1_teamhp_decline p1_teamhp_t10 p1_teamhp_t20
        p1_teamhp_t30 p2_alive_t10 p2_alive_t20
        p2_alive_t30 p2_teamhp_decline p2_teamhp_t10
        p2_teamhp_t20 p2_teamhp_t30 teamhp_decline_diff
        teamhp_diff_t10 teamhp_diff_t20 teamhp_diff_t30
    """.split(),
    'aimilia_flow': """
        dmg_share dmg_surplus dmg_surplus_integral
        dmg_surplus_late heal_diff p1_big_hits
        p1_dmg_taken p1_dmg_taken_early p1_dmg_taken_late
        p1_healed p1_net_hp p1_worst_hit
        p2_big_hits p2_dmg_taken p2_dmg_taken_early
        p2_dmg_taken_late p2_healed p2_net_hp
        p2_worst_hit
    """.split(),
    'aimilia_matchup': """
        diff_eff_mean diff_immune_rate diff_se_rate
        diff_stab_rate p1_eff_late_mean p1_eff_mean
        p1_immune_rate p1_n_damaging p1_resist_rate
        p1_se_rate p1_stab_rate p2_eff_late_mean
        p2_eff_mean p2_immune_rate p2_n_damaging
        p2_resist_rate p2_se_rate p2_stab_rate
    """.split(),
    'aimilia_speed': """
        p1_active_spe_mean p1_priority_rate p1_speed_adv_late
        p1_speed_adv_rate p2_active_spe_mean p2_priority_rate
        speed_gap_mean speed_tie_rate
    """.split(),
    'aimilia_damage': """
        edmg_share edmg_surplus edmg_surplus_late
        p1_edmg_late_mean p1_edmg_max p1_edmg_mean
        p1_edmg_sum p2_edmg_late_mean p2_edmg_max
        p2_edmg_mean p2_edmg_sum
    """.split(),
    'aimilia_hazards': """
        diff_doomed diff_recovery_rate diff_spikes_up
        diff_tox_rate p1_brn_rate p1_doomed
        p1_frz_rate p1_n_haze p1_n_leechseed
        p1_n_phazing p1_n_rapidspin p1_n_recovery
        p1_n_sleep p1_n_species_toxed p1_n_spikes
        p1_n_toxic p1_opp_switchins_after_spikes p1_par_rate
        p1_psn_rate p1_recovery_rate p1_spikes_turn
        p1_spikes_turns_up p1_spikes_up p1_tox_at_end
        p1_tox_rate p2_brn_rate p2_doomed
        p2_frz_rate p2_n_haze p2_n_leechseed
        p2_n_phazing p2_n_rapidspin p2_n_recovery
        p2_n_sleep p2_n_species_toxed p2_n_spikes
        p2_n_toxic p2_opp_switchins_after_spikes p2_par_rate
        p2_psn_rate p2_recovery_rate p2_spikes_turn
        p2_spikes_turns_up p2_spikes_up p2_tox_at_end
        p2_tox_rate
    """.split(),
    'aimilia_moves': """
        diff_can_cure_status diff_n_parmove diff_n_setup
        diff_slp_dead_rate p1_can_cure_status p1_explosion_faint_share
        p1_has_trap_wincon p1_n_batonpass p1_n_confuse
        p1_n_explosion p1_n_healbell p1_n_parmove
        p1_n_perishsong p1_n_rest p1_n_screens
        p1_n_setup p1_n_sleeptalk p1_n_thief
        p1_n_trapping p1_resttalk p1_slp_acted_rate
        p1_slp_dead_rate p1_slp_turns p2_can_cure_status
        p2_explosion_faint_share p2_has_trap_wincon p2_n_batonpass
        p2_n_confuse p2_n_explosion p2_n_healbell
        p2_n_parmove p2_n_perishsong p2_n_rest
        p2_n_screens p2_n_setup p2_n_sleeptalk
        p2_n_thief p2_n_trapping p2_resttalk
        p2_slp_acted_rate p2_slp_dead_rate p2_slp_turns
    """.split(),
    'leonard_battle_state': """
        diff_final_boost_atk diff_final_boost_def diff_final_boost_spa
        diff_final_boost_spd diff_final_boost_spe diff_final_brn
        diff_final_frz diff_final_par diff_final_psn
        diff_final_slp diff_final_tox diff_last_seen_age_mean
        diff_observed_alive diff_observed_hp_sq diff_observed_injured
        diff_recent_10_faints diff_recent_10_hp_gain diff_recent_10_hp_loss
        diff_recent_10_no_move diff_recent_5_faints diff_recent_5_hp_gain
        diff_recent_5_hp_loss diff_recent_5_no_move diff_roster_brn
        diff_roster_frz diff_roster_par diff_roster_psn
        diff_roster_slp diff_roster_tox p1_final_boost_atk
        p1_final_boost_def p1_final_boost_spa p1_final_boost_spd
        p1_final_boost_spe p1_final_brn p1_final_frz
        p1_final_par p1_final_psn p1_final_slp
        p1_final_tox p1_last_seen_age_mean p1_observed_alive
        p1_observed_hp_sq p1_observed_injured p1_recent_10_faints
        p1_recent_10_hp_gain p1_recent_10_hp_loss p1_recent_10_no_move
        p1_recent_5_faints p1_recent_5_hp_gain p1_recent_5_hp_loss
        p1_recent_5_no_move p1_roster_brn p1_roster_frz
        p1_roster_par p1_roster_psn p1_roster_slp
        p1_roster_tox p2_final_boost_atk p2_final_boost_def
        p2_final_boost_spa p2_final_boost_spd p2_final_boost_spe
        p2_final_brn p2_final_frz p2_final_par
        p2_final_psn p2_final_slp p2_final_tox
        p2_last_seen_age_mean p2_observed_alive p2_observed_hp_sq
        p2_observed_injured p2_recent_10_faints p2_recent_10_hp_gain
        p2_recent_10_hp_loss p2_recent_10_no_move p2_recent_5_faints
        p2_recent_5_hp_gain p2_recent_5_hp_loss p2_recent_5_no_move
        p2_roster_brn p2_roster_frz p2_roster_par
        p2_roster_psn p2_roster_slp p2_roster_tox
    """.split(),
    'leonard_remaining_strength': """
        p1_observation_age_max p1_observation_age_mean p1_remaining_atk_max
        p1_remaining_atk_sum p1_remaining_def_max p1_remaining_def_sum
        p1_remaining_hp_max p1_remaining_hp_sum p1_remaining_spa_max
        p1_remaining_spa_sum p1_remaining_spd_max p1_remaining_spd_sum
        p1_remaining_spe_max p1_remaining_spe_sum p1_unseen_members
        p2_known_lead_remaining_atk p2_known_lead_remaining_def p2_known_lead_remaining_hp
        p2_known_lead_remaining_spa p2_known_lead_remaining_spd p2_known_lead_remaining_spe
        p2_observation_age_max p2_observation_age_mean p2_observed_species
        p2_species_without_known_stats
    """.split(),
    'leonard_trajectory': """
        alive_diff_change_last10 alive_diff_change_last5 alive_diff_t15
        alive_diff_t25 alive_diff_t5 p1_alive_change_last10
        p1_alive_change_last5 p1_alive_t15 p1_alive_t25
        p1_alive_t5 p1_new_faints_last10 p1_new_faints_last5
        p1_observed_t10 p1_observed_t15 p1_observed_t20
        p1_observed_t25 p1_observed_t30 p1_observed_t5
        p1_teamhp_change_last10 p1_teamhp_change_last5 p1_teamhp_t15
        p1_teamhp_t25 p1_teamhp_t5 p2_alive_change_last10
        p2_alive_change_last5 p2_alive_t15 p2_alive_t25
        p2_alive_t5 p2_new_faints_last10 p2_new_faints_last5
        p2_observed_t10 p2_observed_t15 p2_observed_t20
        p2_observed_t25 p2_observed_t30 p2_observed_t5
        p2_teamhp_change_last10 p2_teamhp_change_last5 p2_teamhp_t15
        p2_teamhp_t25 p2_teamhp_t5 teamhp_diff_change_last10
        teamhp_diff_change_last5 teamhp_diff_t15 teamhp_diff_t25
        teamhp_diff_t5
    """.split(),
    'leonard_boost_threat': """
        diff_active_atk_brn_interaction diff_active_atk_hp_stage diff_active_atk_par_interaction
        diff_active_atk_slp_interaction diff_active_atk_stage diff_active_spa_brn_interaction
        diff_active_spa_hp_stage diff_active_spa_par_interaction diff_active_spa_slp_interaction
        diff_active_spa_stage diff_active_spe_brn_interaction diff_active_spe_hp_stage
        diff_active_spe_par_interaction diff_active_spe_slp_interaction diff_active_spe_stage
        diff_observed_atk_threat_max diff_observed_atk_threat_sum diff_observed_spa_threat_max
        diff_observed_spa_threat_sum diff_observed_spe_threat_max diff_observed_spe_threat_sum
        diff_recent_atk_threat_max diff_recent_spa_threat_max diff_recent_spe_threat_max
        p1_active_atk_brn_interaction p1_active_atk_hp_stage p1_active_atk_par_interaction
        p1_active_atk_slp_interaction p1_active_atk_stage p1_active_spa_brn_interaction
        p1_active_spa_hp_stage p1_active_spa_par_interaction p1_active_spa_slp_interaction
        p1_active_spa_stage p1_active_spe_brn_interaction p1_active_spe_hp_stage
        p1_active_spe_par_interaction p1_active_spe_slp_interaction p1_active_spe_stage
        p1_observed_atk_threat_max p1_observed_atk_threat_sum p1_observed_spa_threat_max
        p1_observed_spa_threat_sum p1_observed_spe_threat_max p1_observed_spe_threat_sum
        p1_recent_atk_threat_max p1_recent_spa_threat_max p1_recent_spe_threat_max
        p2_active_atk_brn_interaction p2_active_atk_hp_stage p2_active_atk_par_interaction
        p2_active_atk_slp_interaction p2_active_atk_stage p2_active_spa_brn_interaction
        p2_active_spa_hp_stage p2_active_spa_par_interaction p2_active_spa_slp_interaction
        p2_active_spa_stage p2_active_spe_brn_interaction p2_active_spe_hp_stage
        p2_active_spe_par_interaction p2_active_spe_slp_interaction p2_active_spe_stage
        p2_observed_atk_threat_max p2_observed_atk_threat_sum p2_observed_spa_threat_max
        p2_observed_spa_threat_sum p2_observed_spe_threat_max p2_observed_spe_threat_sum
        p2_recent_atk_threat_max p2_recent_spa_threat_max p2_recent_spe_threat_max
    """.split(),
}

_WORDS = {
    'atk':'attack', 'def':'defense', 'spa':'special_attack',
    'spd':'special_defense', 'spe':'speed', 'brn':'burn', 'frz':'freeze',
    'par':'paralysis', 'psn':'poison', 'slp':'sleep', 'tox':'toxic_poison',
    'leechseed':'leech_seed', 'sleeptalk':'sleep_talk', 'healbell':'team_status_cure',
    'batonpass':'baton_pass', 'perishsong':'perish_song', 'rapidspin':'rapid_spin',
    'destinybond':'destiny_bond', 'resttalk':'rest_and_sleep_talk_observed',
    'hppct':'active_hp_fraction', 'teamhp':'presumed_team_hp_bars',
    'edmg':'expected_damage_proxy', 'eff':'effect', 'se':'super_effective',
    'std':'standard_deviation', 'opp':'opponent', 'ko':'faint',
}


def _readable(key):
    key = re.sub(r'^diff_', 'p1_minus_p2_', key)
    key = re.sub(r'^gap_', 'p1_minus_p2_', key)
    key = re.sub(r'^alive_diff_', 'p1_minus_p2_presumed_alive_', key)
    key = re.sub(r'^teamhp_diff_', 'p1_minus_p2_teamhp_', key)
    key = key.replace('teamhp_decline_diff','p1_minus_p2_teamhp_decline')
    key = re.sub(r'_t(\d+)_(\d+)', lambda m:f'_turns{int(m[1]):02d}_{int(m[2]):02d}', key)
    key = re.sub(r'_t(\d+)(?=_|$)', r'_turn\1', key)
    key = key.replace('_at_turn30','_turn30')
    key = re.sub(r'(^|_)n_', r'\1count_', key)
    key = key.replace('_no_of_moves', '_recorded_move_count')
    key = key.replace('_rate', '_fraction')
    return '_'.join(_WORDS.get(part,part) for part in key.split('_'))


def _core_name(key):
    if key == 'first_ko_advantage':
        return 'events_first_faint_time_weighted_advantage'
    if key == 'alive_diff_at_turn30':
        return 'resources_p1_minus_p2_presumed_alive_turn30'
    for side in ('p1','p2'):
        suffix = key.removeprefix(side+'_')
        if suffix == key:
            continue
        names = {
            'dmg':f'hp_{side}_observed_damage_dealt_total_full_entry_baseline',
            'heal':f'hp_{side}_observed_healing_total_full_entry_baseline',
            'alive_at_turn30':f'resources_{side}_presumed_alive_turn30',
            'damage_surplus':f'hp_{side}_positive_net_damage_advantage_mean_turns01_30',
            'team_hp_pct_at_turn30':f'resources_{side}_presumed_team_hp_percent_turn30',
            'switch':f'transitions_{side}_active_species_change_count',
            'no_of_moves':f'moves_{side}_recorded_count',
            'special_moves':f'moves_{side}_special_count',
            'no_damage_moves':f'moves_{side}_no_adjacent_target_hp_loss_count',
            'max_observed_move_damage':f'moves_{side}_largest_adjacent_target_hp_drop',
            'fainting_turns':f'events_{side}_distinct_hp_or_status_fainted_species_count',
            'mean_boosts':f'boosts_{side}_active_stage_sum_mean',
            'boost_sum_at_turn30':f'boosts_{side}_active_stage_sum_turn30',
            'positive_boosts_at_turn30':f'boosts_{side}_active_positive_stage_sum_turn30',
            'negative_boosts_at_turn30':f'boosts_{side}_active_negative_stage_signed_sum_turn30',
            'healthy_living_members':f'resources_{side}_presumed_healthy_alive_members_hp_above_half',
            'remaining_special_bulk':f'resources_{side}_retained_hp_times_base_special_bulk_sum',
            'hp_weighted_base_special_attack':f'resources_{side}_retained_hp_weighted_base_special_attack_mean',
            'leechseed_rate':f'effects_{side}_leech_seed_observed_turn_fraction',
            'team_diversity':f'roster_{side}_supplied_distinct_type_count',
        }
        if suffix in names:
            return names[suffix]
        if suffix.startswith('mean_'):
            stat = _WORDS.get(suffix[5:],suffix[5:])
            population = 'supplied' if side == 'p1' else 'revealed_species'
            return f'roster_{side}_{population}_base_{stat}_mean'
        if suffix.startswith('lead_'):
            return f'roster_{side}_supplied_lead_base_{_readable(suffix[5:])}'
    raise KeyError(key)


def _name(group, key):
    value = _readable(key)
    if group in ('selected','expanded_core'):
        return _core_name(key)
    if group == 'roster_shape':
        return 'roster_shape_' + value
    if group == 'status_resources':
        return 'status_resources_' + value
    if group == 'recent_flow':
        return 'recent_member_flow_' + value
    if group == 'active_position':
        return 'active_position_' + value
    if group == 'aimilia_static':
        if key.startswith('diff_'):
            stat, aggregate = key[5:].rsplit('_',1)
            return f'roster_level_scaled_p1_team_{_readable(stat)}_{aggregate}_minus_p2_lead'
        return 'roster_level_scaled_' + value
    if group == 'aimilia_timeline':
        prefix = ('active_hp_' if 'hppct' in key else 'status_exposure_' if
                  ('status' in key or 'slp' in key) else 'move_usage_' if
                  ('move' in key or 'accuracy' in key or 'base_power' in key) else
                  'active_transitions_')
        value = value.replace('_faints','_distinct_status_fainted_species_count')
        value = value.replace('_switches','_active_species_change_count')
        value = value.replace('_count_species_seen','_distinct_active_species_count')
        return prefix + value
    if group == 'aimilia_effects':
        return ('boost_history_' if 'boost' in key else 'effect_exposure_') + value
    if group == 'aimilia_roster':
        return 'team_resources_' + value
    if group == 'aimilia_flow':
        value = {
            'dmg_surplus':'p1_minus_p2_observed_damage_dealt',
            'dmg_surplus_late':'p1_minus_p2_observed_damage_dealt_turns21_30',
            'dmg_surplus_integral':'p1_minus_p2_cumulative_damage_dealt_mean',
            'dmg_share':'p1_observed_damage_dealt_share',
            'heal_diff':'p1_minus_p2_healing_total',
        }.get(key,value)
        value = value.replace('dmg_taken','received_hp_loss').replace('dmg','damage')
        return 'hp_exchange_full_entry_' + value
    if group == 'aimilia_matchup':
        return 'move_matchups_' + value.replace('effect_', 'effectiveness_')
    if group == 'aimilia_speed':
        value = 'p1_minus_p2_adjusted_speed_mean' if key=='speed_gap_mean' else value
        return 'speed_control_' + value
    if group == 'aimilia_damage':
        value = {'edmg_surplus':'p1_minus_p2_expected_damage_proxy_sum',
                 'edmg_surplus_late':'p1_minus_p2_expected_damage_proxy_mean_turns21_30',
                 'edmg_share':'p1_expected_damage_proxy_share'}.get(key,value)
        return 'attack_pressure_' + value
    if group == 'aimilia_hazards':
        replacements = {
            'diff_doomed':'p2_minus_p1_observed_toxic_or_seeded_alive_count',
            'diff_tox_rate':'p2_minus_p1_active_toxic_poison_fraction',
        }
        value = _readable(replacements.get(key,key))
        value = value.replace('_doomed','_observed_toxic_or_seeded_alive_count')
        value = value.replace('_spikes_up','_spikes_use_observed')
        value = value.replace('_spikes_turns_up','_turns_since_first_spikes_use')
        value = value.replace('_spikes_turn','_first_spikes_use_turn_or31')
        value = value.replace('_tox_at_end','_active_toxic_poison_last_observed')
        return ('hazard_use_' if 'spikes' in key or 'phazing' in key or 'rapidspin' in key
                else 'status_pressure_') + value
    if group == 'aimilia_moves':
        value = _readable('p2_minus_p1_sleep_no_action_fraction' if key=='diff_slp_dead_rate' else key)
        value = value.replace('_sleep_dead_fraction','_sleep_no_action_turn_fraction')
        value = value.replace('_sleep_acted_fraction','_action_fraction_while_asleep')
        value = value.replace('_has_trap_wincon','_trapping_and_perish_song_observed')
        value = value.replace('_can_cure_status','_team_status_cure_use_observed')
        value = value.replace('_parmove','_paralysis_inducing_move')
        return 'move_patterns_' + value
    if group == 'leonard_battle_state':
        value = re.sub(r'_recent_(5|10)_',r'_last\1_observations_',value)
        value = value.replace('_no_move','_no_recorded_move_fraction')
        value = value.replace('_observed_hp_sq','_observed_hp_fraction_squared_sum')
        value = value.replace('_observed_injured','_observed_alive_below_half_hp_count')
        value = value.replace('_observed_alive','_observed_alive_count')
        return 'last_seen_state_' + value
    if group == 'leonard_remaining_strength':
        return 'retained_strength_' + value
    if group == 'leonard_trajectory':
        return 'team_trajectory_' + value
    if group == 'leonard_boost_threat':
        return 'boost_threat_' + value
    raise KeyError(group)


_GROUP_NOTES = {
    'selected':'Observed turns 1–30; team health proxies treat unseen roster slots as full HP.',
    'expanded_core':'Observed turns 1–30 or supplied initial roster/lead metadata; no target input.',
    'roster_shape':'Supplied P1 roster or P2 lead metadata; base stats are species values before level scaling.',
    'status_resources':'Last-seen supplied/revealed members only; unknown P2 slots contribute no inferred HP or status.',
    'recent_flow':'Within the named actual-turn window; HP changes use a prior known member observation or supplied initial full-HP baseline.',
    'active_position':'Last observed active state; supplied metadata only; unavailable opponent stat proxies stay missing.',
    'aimilia_static':'Initial roster/lead base stats multiplied by level/100 where the measurement is a stat.',
    'aimilia_timeline':'Active-state observations in turns 1–30; early is turns 1–10 and late is turns 21–30.',
    'aimilia_effects':'Fraction of observed turns with an effect, or summary of recorded active boost stages.',
    'aimilia_roster':'Latest-seen member HP at the named checkpoint; unseen slots in a six-member roster are full HP/alive.',
    'aimilia_flow':'Per-member HP changes from a full-HP first-entry baseline; losses are observations, not attributed attacks.',
    'aimilia_matchup':'Recorded damaging moves with resolvable per-battle species metadata; unknown matchups are excluded.',
    'aimilia_speed':'Level/boost/paralysis-adjusted base-speed proxy; comparison rates use resolvable turns only.',
    'aimilia_damage':'Simplified move power × accuracy × type multiplier × STAB × offense/defense, divided by 100; not exact damage.',
    'aimilia_hazards':'Recorded move usage and observed status only; Spikes use does not establish a currently active hazard.',
    'aimilia_moves':'Recorded move-category use and observed sleep/action patterns in turns 1–30.',
    'leonard_battle_state':'Observed species only; last-seen bench HP/status may be stale. Recent HP deltas compare adjacent same-active species.',
    'leonard_remaining_strength':'HP-weighted level-scaled supplied stats; unknown enemy species receive no inferred stats.',
    'leonard_trajectory':'Checkpoint latest-seen HP/alive proxy; unrevealed slots are presumed healthy; changes are later minus earlier.',
    'leonard_boost_threat':'Observed boost-stage/HP/status interactions; positive-stage threats use last-seen living members and may be stale.',
}


def _units(group, key, name):
    if 'hp_weighted' in key and key.endswith('attack_proxy'):
        return 'adjusted base-stat proxy × HP fraction'
    if key.endswith('remaining_special_bulk'):
        return 'base HP × base special defense × retained HP fraction'
    if key.endswith('max_observed_move_damage'):
        return 'HP fraction'
    if key.endswith(('_rows','_slots','_diversity','_unseen_members','_observed_species','_species_without_known_stats')):
        return 'count'
    if key == 'first_ko_advantage':
        return 'signed normalized turn weight'
    if key.endswith('hp_weighted_base_special_attack'):
        return 'base special attack'
    if group == 'aimilia_static':
        if 'level' in key:
            return 'level'
        if any(stat in key.split('_') for stat in ('atk','def','spa','spd','spe','hp')):
            return 'level-scaled base stat'
        return 'count'
    if group == 'roster_shape':
        if 'fraction' in key:
            return 'fraction of known items'
        if 'level' in key:
            return 'level'
        if 'bulk' in key:
            return 'base HP × base defense'
        if 'ratio' in key or 'special_to_physical' in key:
            return 'stat ratio'
        if 'count' in key or 'members' in key:
            return 'count'
        return 'base stat'
    if group == 'aimilia_speed':
        return 'adjusted base-speed proxy' if 'spe_mean' in key or 'gap_mean' in key else 'fraction'
    if group == 'aimilia_effects':
        return 'boost stages' if 'boost' in key else 'fraction of observed turns'
    if group == 'aimilia_roster':
        return 'HP bars' if 'teamhp' in key else 'count'
    if group == 'aimilia_flow':
        return 'fraction' if 'share' in key else 'count' if 'big_hits' in key else 'HP bars'
    if group == 'aimilia_matchup':
        return 'type multiplier' if ('eff_mean' in key or 'eff_late_mean' in key) else 'count' if 'n_damaging' in key else 'fraction'
    if group == 'aimilia_damage':
        return 'fraction' if 'share' in key else 'damage-proxy units'
    if group == 'leonard_trajectory':
        return 'HP bars' if 'teamhp' in key else 'count'
    if group == 'leonard_battle_state':
        if 'final_boost' in key:
            return 'boost stages'
        if 'final_' in key:
            return 'indicator' if not key.startswith('diff_') else 'signed indicator difference'
        if 'age' in key:
            return 'turns'
        if 'hp_sq' in key:
            return 'squared HP fraction'
        if 'no_move' in key:
            return 'fraction'
        if 'hp_loss' in key or 'hp_gain' in key:
            return 'HP bars'
        return 'count'
    if group == 'active_position' and key.endswith('stats_supplied'):
        return 'indicator'
    if group == 'aimilia_moves':
        if key.endswith('rate') or key.endswith('share'):
            return 'fraction'
        if any(token in key for token in ('can_cure_status','has_trap_wincon','resttalk')):
            return 'indicator' if not key.startswith('diff_') else 'signed indicator difference'
        return 'turns' if key.endswith('slp_turns') else 'count'
    if group == 'aimilia_hazards':
        if key.endswith('rate'):
            return 'fraction'
        if 'spikes_turn' in key:
            return 'turns'
        if key.endswith('spikes_up') or key.endswith('tox_at_end'):
            return 'indicator' if not key.startswith('diff_') else 'signed indicator difference'
        return 'count'
    if 'team_hp_percent' in name:
        return 'percent of six full HP bars'
    if 'hp_sq' in key:
        return 'squared HP fraction'
    if 'rate' in key or 'fraction' in name or 'share' in key:
        return 'fraction'
    if any(x in key for x in ('observation_age','last_seen_age','spikes_turn')) or key in ('final_observed_turn','observed_turn_count'):
        return 'turns'
    if group == 'aimilia_damage':
        return 'damage-proxy units'
    if group == 'aimilia_matchup' and ('eff' in key):
        return 'type multiplier'
    if 'hp_stage' in key or ('threat' in key and group=='leonard_boost_threat') or 'interaction' in key:
        return 'HP fraction × boost stage'
    if 'boost' in key or key.endswith('_stage'):
        return 'boost stages'
    if 'bulk' in key:
        return 'base HP × base defense × retained HP fraction' if 'remaining' in key else 'base HP × base defense'
    if group == 'leonard_remaining_strength' and 'remaining_' in key:
        return 'level-scaled base stat × retained HP fraction'
    if 'speed_proxy' in key or 'attack_proxy' in key:
        return 'adjusted base-stat proxy'
    if 'hp' in key and not any(x in key for x in ('base_hp','lead_hp','mean_hp')):
        return 'HP bars' if any(x in key for x in ('teamhp','retained','gain','loss')) else 'HP fraction'
    if any(x in key for x in ('dmg','heal','surplus','net_hp','worst_hit')):
        return 'HP bars'
    if 'level' in key:
        return 'level'
    if 'accuracy' in key:
        return 'fraction'
    if 'base_power' in key:
        return 'move base power'
    if any(x in key for x in ('alive','count','members','faint','switch','species','moves')) or re.search(r'(^|_)n_',key):
        return 'count'
    if any(x in key for x in ('observed','supplied','resttalk','spikes_up','tox_at_end','can_cure','wincon')):
        return 'indicator or count'
    if any(stat in key.split('_') for stat in ('atk','def','spa','spd','spe','hp')):
        return 'level-scaled base stat' if group=='aimilia_static' else 'base stat'
    return 'numeric proxy'


def _category(name):
    categories = (
        ('roster_', 'Initial roster'), ('resources_', 'Remaining resources'),
        ('hp_exchange_', 'Observed HP exchange'), ('hp_', 'Observed HP exchange'),
        ('events_', 'Fainting events'), ('transitions_', 'Switching'),
        ('moves_', 'Recorded moves'), ('boosts_', 'Boost stages'),
        ('effects_', 'Effect exposure'), ('status_resources_', 'Retained status resources'),
        ('recent_member_flow_', 'Recent battle flow'), ('active_position_', 'Active position'),
        ('active_hp_', 'Active HP trajectory'), ('status_exposure_', 'Status exposure'),
        ('move_usage_', 'Recorded move usage'), ('active_transitions_', 'Switching and faints'),
        ('boost_history_', 'Boost trajectory'), ('effect_exposure_', 'Effect exposure'),
        ('team_resources_', 'Team resource checkpoints'), ('move_matchups_', 'Move effectiveness'),
        ('speed_control_', 'Speed control'), ('attack_pressure_', 'Estimated attack pressure'),
        ('hazard_use_', 'Hazard usage'), ('status_pressure_', 'Status and recovery'),
        ('move_patterns_', 'Move patterns'), ('last_seen_state_', 'Last-seen state'),
        ('retained_strength_', 'Retained strength'), ('team_trajectory_', 'Team resource trajectory'),
        ('boost_threat_', 'Boost and HP interactions'),
    )
    return next(category for prefix,category in categories if name.startswith(prefix))


def _description(group, key, name):
    words = name.replace('_',' ')
    words = re.sub(r'\bp1\b','P1',words)
    words = re.sub(r'\bp2\b','P2',words)
    words = re.sub(r'\bhp\b','HP',words)
    words = re.sub(r'\bstab\b','same-type attack bonus',words)
    words = words[0].upper() + words[1:] + '.'
    if group == 'aimilia_hazards' and key.endswith('spikes_turn'):
        words += ' Uses 31 when no Spikes use is observed.'
    if key == 'first_ko_advantage':
        words = 'First faint advantage: +(31−turn)/30 for P2 first faint, the negative for P1; simultaneous/no faint is zero.'
    if key.endswith('healthy_living_members'):
        words = 'Presumed living roster members with last-known HP strictly above 0.5 and status exactly healthy, including unseen slots.'
    if group == 'status_resources' and 'retained_poison_hp' in key:
        words += ' Includes both ordinary and toxic poison.'
    if group == 'status_resources' and 'retained_incapacitated_hp' in key:
        words += ' Incapacitated means asleep or frozen.'
    return words + ' ' + _GROUP_NOTES[group]


FEATURE_NAMES = {f'{group}__{key}':_name(group,key)
                 for group,keys in _RAW_FEATURES.items() for key in keys}
if len(FEATURE_NAMES) != len(set(FEATURE_NAMES.values())):
    raise ValueError('Semantic feature names must be unique.')
FEATURE_DOCUMENTATION = {}
for _group,_keys in _RAW_FEATURES.items():
    for _key in _keys:
        _semantic = FEATURE_NAMES[f'{_group}__{_key}']
        FEATURE_DOCUMENTATION[_semantic] = {
            'category':_category(_semantic),
            'description':_description(_group,_key,_semantic),
            'units':_units(_group,_key,_semantic),
        }


def rename_frame(frame):
    """Return a semantic-name copy, preserving order and refusing unknown keys."""
    unknown = set(frame.columns) - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f'Unknown extraction keys: {sorted(unknown)[:5]}')
    return frame.rename(columns=FEATURE_NAMES).copy()


def feature_dictionary(columns=None):
    """Return readable definitions for semantic feature names in requested order."""
    import pandas as pd
    columns = list(FEATURE_DOCUMENTATION) if columns is None else list(columns)
    unknown = set(columns) - set(FEATURE_DOCUMENTATION)
    if unknown:
        raise ValueError(f'Unknown semantic names: {sorted(unknown)[:5]}')
    return pd.DataFrame([{'feature':name, **FEATURE_DOCUMENTATION[name]} for name in columns])
