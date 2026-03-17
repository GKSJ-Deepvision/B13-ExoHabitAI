def validate_input(data):
    required_fields = [
        'planet_radius',
        'orbital_period',
        'equilibrium_temperature',
        'semi_major_axis',
        'stellar_luminosity',
        'stellar_mass'
    ]
    for field in required_fields:
        if field not in data:
            return False, f"Missing field: '{field}'"
    for field in required_fields:
        try:
            float(data[field])
        except (ValueError, TypeError):
            return False, f"'{field}' must be a number"
    return True, None


def calculate_habitability(data):
    pl_rade    = float(data['planet_radius'])
    pl_orbper  = float(data['orbital_period'])
    pl_eqt     = float(data['equilibrium_temperature'])
    pl_orbsmax = float(data['semi_major_axis'])
    st_lum     = float(data['stellar_luminosity'])
    st_mass    = float(data['stellar_mass'])

    # ── Temperature Score (35% weight) ──────────────────────
    # Liquid water zone: 200–320K is ideal
    if 220 <= pl_eqt <= 300:
        temp_score = 100   # Perfect — Earth-like
    elif 180 <= pl_eqt <= 220 or 300 <= pl_eqt <= 340:
        temp_score = 70    # Borderline habitable
    elif 150 <= pl_eqt <= 180 or 340 <= pl_eqt <= 400:
        temp_score = 35    # Too cold/hot
    elif 100 <= pl_eqt <= 150 or 400 <= pl_eqt <= 600:
        temp_score = 15    # Very unlikely
    else:
        temp_score = 2     # Extreme — no chance

    # ── Radius Score (25% weight) ────────────────────────────
    # Earth-size planets more likely to be rocky
    if 0.8 <= pl_rade <= 1.5:
        radius_score = 100  # Earth-like size
    elif 0.5 <= pl_rade <= 0.8 or 1.5 <= pl_rade <= 2.0:
        radius_score = 70   # Close to Earth
    elif 0.3 <= pl_rade <= 0.5 or 2.0 <= pl_rade <= 2.5:
        radius_score = 35   # Too small/big
    elif pl_rade <= 0.3 or 2.5 <= pl_rade <= 4.0:
        radius_score = 15   # Very unlikely
    else:
        radius_score = 2    # Gas giant — no surface

    # ── Luminosity Score (20% weight) ───────────────────────
    # Sun-like stars more stable
    if 0.5 <= st_lum <= 1.5:
        lum_score = 100     # Sun-like
    elif 0.1 <= st_lum <= 0.5 or 1.5 <= st_lum <= 3.0:
        lum_score = 55      # Dim/bright stars
    elif 0.01 <= st_lum <= 0.1 or 3.0 <= st_lum <= 10:
        lum_score = 20      # Very dim/very bright
    else:
        lum_score = 5       # Extreme

    # ── Semi-Major Axis Score (10% weight) ──────────────────
    # Habitable zone: 0.7–1.5 AU for Sun-like
    if 0.7 <= pl_orbsmax <= 1.4:
        axis_score = 100    # Classic habitable zone
    elif 0.5 <= pl_orbsmax <= 0.7 or 1.4 <= pl_orbsmax <= 2.0:
        axis_score = 55     # Edge of habitable zone
    elif 0.3 <= pl_orbsmax <= 0.5 or 2.0 <= pl_orbsmax <= 3.0:
        axis_score = 20     # Outside habitable zone
    else:
        axis_score = 5      # Far outside

    # ── Orbital Period Score (5% weight) ────────────────────
    if 200 <= pl_orbper <= 500:
        period_score = 100
    elif 100 <= pl_orbper <= 200 or 500 <= pl_orbper <= 700:
        period_score = 65
    elif 50 <= pl_orbper <= 100 or 700 <= pl_orbper <= 1000:
        period_score = 30
    else:
        period_score = 10

    # ── Stellar Mass Score (5% weight) ──────────────────────
    if 0.7 <= st_mass <= 1.3:
        mass_score = 100    # Sun-like mass
    elif 0.4 <= st_mass <= 0.7 or 1.3 <= st_mass <= 2.0:
        mass_score = 55
    elif 0.1 <= st_mass <= 0.4 or 2.0 <= st_mass <= 5.0:
        mass_score = 20
    else:
        mass_score = 5

    # ── Final Weighted Score ─────────────────────────────────
    final_score = (
        temp_score   * 0.35 +
        radius_score * 0.25 +
        lum_score    * 0.20 +
        axis_score   * 0.10 +
        period_score * 0.05 +
        mass_score   * 0.05
    )

    # ── Penalty System ───────────────────────────────────────
    # Extreme temperature penalty
    if pl_eqt > 500 or pl_eqt < 100:
        final_score *= 0.3   # 70% penalty

    # Gas giant penalty
    if pl_rade > 4.0:
        final_score *= 0.2   # 80% penalty

    # Too close to star penalty
    if pl_orbsmax < 0.1:
        final_score *= 0.25  # 75% penalty

    final_score = round(min(final_score, 100), 2)
    prediction  = 1 if final_score >= 50 else 0

    return final_score, prediction


def get_confidence_label(score):
    if score >= 85:
        return "Highly Habitable"
    elif score >= 70:
        return "Moderately Habitable"
    elif score >= 50:
        return "Possibly Habitable"
    elif score >= 30:
        return "Unlikely Habitable"
    elif score >= 15:
        return "Very Unlikely"
    else:
        return "Not Habitable"