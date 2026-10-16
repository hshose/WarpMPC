"""Nominal table-tennis equipment, with sources in docs/equipment_specs.md.

The ball values are ITTF specifications. Racket dimensions vary; the reference
head is the common 157 x 150 mm Butterfly Primorac size. Stage 1 approximates
that head by a disk whose diameter equals the reference width.
"""

BALL_MASS_KG = 0.0027
BALL_RADIUS_M = 0.020
RACKET_HEAD_LENGTH_M = 0.157
RACKET_HEAD_WIDTH_M = 0.150
PADDLE_RADIUS_M = RACKET_HEAD_WIDTH_M / 2

EQUIPMENT_REFERENCE = {
    'ball_standard': 'ITTF Laws 2.3.1–2.3.3: plastic, 40 mm diameter, 2.7 g',
    'ball_source': 'https://db.ittf.com/sites/default/files/public/2026-02/2026_Statutes_v1_consolidated_clean.pdf',
    'racket_reference': 'Butterfly Primorac, 157 x 150 mm head',
    'racket_source': 'https://shop.butterflyonline.com/primorac-fl',
    'racket_model': '75 mm-radius circular approximation; not a regulation-mandated size',
}
