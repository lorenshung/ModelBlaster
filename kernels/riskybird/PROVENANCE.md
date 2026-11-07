# Imported RiskyBird contract files (verbatim copies; do not edit here)

| file | source | sha256 |
|---|---|---|
| b_driver.h | riskybird soc/phase2/b_unit/b_driver.h (CONTRACT.md v1) | 8833854b528b925a33cad1a65fbde5281e1d01c633629f75b4815cfee1c3e6e2 |
| b_model.c | riskybird soc/phase2/b_unit/b_model.c | 5d5f5d81f699d60c7202997a093f71cbb9dcb1a3e551bdd87efd08babadfc720 |

dronet_conv0_driver.c.in is ModelBlaster's own (the conv0 engine driver template).
| d_driver.h | riskybird soc/phase2/d_engine/d_driver.h (SPEC.md §5-7, contract frozen 2026-11-07) | a6236db75cfe013ee7230a1c9524eb28e84f4969154f83b7fbdbde92d8c68d2a |
| d_model.c | riskybird soc/phase2/d_engine/d_model.c (host functional model of D) | c61bc3e3530c11d610c8b3890ad4d434313feb7a26f3faf05a523ae97a29b144 |
| ../../pipeline/riskybird_d/pack_d.py | riskybird soc/phase2/d_engine/pack_d.py (reference descriptor/weight packer) | cf0affe2ebe32a265f561a5cf3e0e4aec2d2e81c93c01473fbb92f7bf5dbed57 |

d_engine_host.h is ModelBlaster's: it wraps d_model.c (main renamed) into a per-frame call.
