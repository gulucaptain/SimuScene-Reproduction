# Paper-to-code coverage

| Paper component | Code | Fidelity / missing information |
|---|---|---|
| 3.1 decomposition | external prepared JSON | not reproduced; base model CUDA dependencies |
| 3.2 pose alignment / A.2 | alignment.py | visible input required; Powell instead of Adam; returns candidates; no silhouette scoring/FoundationPose |
| 3.3 / A.3 diagnostic physics | core.py | MuJoCo CPU; paper dt/friction/contact parameters; independent collision importer |
| 3.4 / A.4 stretching | core.stretch | signed displacement, canonical axis, fixed anti-gravity face; no resampling model |
| 3.4 FM-DPO | callback interface only | no trained adapters or training pairs; not implemented/trained |
| A.5 composition | core.correct / settle | bottom-first + simultaneous settling; no undocumented final rollback threshold guessed |
| B.1 metrics | core.evaluate / experiment | AABB IoU, penetration >2mm, drift <5cm; all-object and eligible subset separately; no image IoU |
| Paper main tables | none | original scenes, upstream models and baselines not run |
| Downstream manipulation / humanoid | none | not reproduced |

## Assumptions and limitations
- MuJoCo 4.0.1 is our recorded version; no original simulator version has been established.
- World convention is Z-up. Ordering uses lowest world Z to avoid gravity-sign ambiguity.
- Non-box meshes use canonical OBB dimensions. Convex inputs use direct hulls; concave inputs use V-HACD with local settings maxConvexHulls=16 and resolution=100000, as the original decomposition settings have not been established.
- No-contact timeout, opposed contact normals and anchored-penetration failures remain explicit; no ground-truth fallback.
- Physical pose error and shape error are not identifiable from a fall distance alone. The pose-error suite quantifies this issue.
- Source paper and hash are saved privately in `sources/`. Check redistribution licenses before publishing paper assets. No publication has been performed.
