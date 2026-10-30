"""
PAXRay++ class-based keypoint schema (instance-wise), mirroring SynthFB.
Two view-specific schemas: SCHEMA_FRONTAL and SCHEMA_LATERAL.

Design principles:

  1. INSTANCE-WISE. Keypoints live on each annotation / mask instance under
     that instance's own category schema (like SynthFB device annotations).
     No global / free-floating landmark points.

  2. NAMED KEYPOINT CLASSES + ANATOMICAL SIDE. Single-side classes carry
     `side` at the class level (side-less names); combined/bilateral classes
     use `_left`/`_right` suffixes.

  3. ANGLES ARE DECOMPOSED ACROSS MASKS WITH ROLE-QUALIFIED NAMES:
         {landmark}_{role}[ _{side} ]
     role = host structure (diaphragmatic/pulmonary/cardiac/costal/vertebral/
     sternal/clavicular/...). Each decomposed point carries `group` = shared
     landmark id (e.g. "costophrenic_left") so the limbs re-link.

  4. DERIVED FROM MASKS, NOT GENERATED. Every kpt is computed deterministically
     from the existing RLE masks.

IMPORTANT (learned from the real masks):
  * Masks of anatomically adjacent structures DO NOT TOUCH (lung->hemidiaphragm
    gap is 19-136 px). So there is NO `junction = intersect` method. The two
    valid relational primitives are:
        - directed single-mask extreme  (the corner where THIS structure forms
          the angle, e.g. lung infero-lateral corner = costophrenic angle); and
        - `nearest` (the point on THIS mask closest to a partner mask, for true
          joints like sterno-clavicular / costo-vertebral).
    Decomposed limbs are therefore DISTINCT nearby points -> `group` means
    "same anatomical angle", not "same pixel".
  * Lateral images use a different, reduced label set (no left/right lung, no
    hemidiaphragms); SCHEMA_LATERAL is built on an ANTERIOR-POSTERIOR axis
    (anterior = toward sternum, posterior = toward spine).

COCO output: keypoints (x, y, v), v in {0,1,2} (0 absent/off-frame,
1 present-but-occluded, 2 visible). `group` is extra metadata (not stock COCO).
"""

from dataclasses import dataclass, field
from typing import List, Tuple


# --------------------------------------------------------------------------- #
# Keypoint descriptor
# --------------------------------------------------------------------------- #
# method vocabulary (how to compute the point from the mask(s)):
#  FRONTAL directed extremes (axis = midline; lateral = away from spine):
#    extreme:top | extreme:bottom
#    extreme:lateral | extreme:medial
#    extreme:infero_lateral | extreme:infero_medial
#    extreme:supero_lateral | extreme:supero_medial
#  LATERAL directed extremes (axis = sternum[anterior] <-> spine[posterior]):
#    extreme:top | extreme:bottom
#    extreme:anterior | extreme:posterior
#    extreme:antero_superior | extreme:antero_inferior
#    extreme:postero_superior | extreme:postero_inferior
#  shared:
#    centroid                 mask centre of mass (largest component)
#    corner:SL|SR|IL|IR       frontal minAreaRect corner (sup/inf, left/right)
#    corner:AS|AI|PS|PI       lateral minAreaRect corner (ant/post, sup/inf)
#    skel:top|mid|bottom      skeletonize -> path endpoint / midpoint
#    centerline:F             point at arc-length fraction F (0..1) along the
#                             medial-axis centerline (tubular vessels/airway)
#    curv:centerline          max-curvature point along the centerline (sternal
#                             angle = manubriosternal bend; robust)
#    curv:max                 sharpest-curvature point on the raw contour (legacy)
#    nearest                  point on THIS mask closest to via[0]'s mask
#    overlap_centroid         centroid of (this mask  &  via[0]'s mask)
#    todo                     desirable but not derivable from current masks
@dataclass(frozen=True)
class KP:
    name: str
    method: str
    via: Tuple[str, ...] = ()        # partner classes (computed against / related to)
    group: str = ""                  # shared landmark id for decomposed angles
    view: str = "both"               # both | frontal | lateral


@dataclass
class ClassSchema:
    side: str                        # left | right | both | midline | none
    keypoints: List[KP] = field(default_factory=list)
    skeleton: List[Tuple[str, str]] = field(default_factory=list)

    def names(self) -> List[str]:
        return [k.name for k in self.keypoints]

    def coco_skeleton(self) -> List[List[int]]:
        idx = {k.name: i + 1 for i, k in enumerate(self.keypoints)}
        return [[idx[a], idx[b]] for a, b in self.skeleton if a in idx and b in idx]


# --------------------------------------------------------------------------- #
# PAXRay++ categories (id, name)  -- exact strings (trailing spaces on
# "left pulmonary artery " / "right pulmonary artery " are part of the name)
# --------------------------------------------------------------------------- #
CATEGORIES = [
    (0, "spine"), (1, "cervical spine"), (2, "thoracic spine"), (3, "lumbar spine"),
    (4, "vertebrae C1"), (5, "vertebrae C2"), (6, "vertebrae C3"), (7, "vertebrae C4"),
    (8, "vertebrae C5"), (9, "vertebrae C6"), (10, "vertebrae C7"),
    (11, "vertebrae T1"), (12, "vertebrae T2"), (13, "vertebrae T3"), (14, "vertebrae T4"),
    (15, "vertebrae T5"), (16, "vertebrae T6"), (17, "vertebrae T7"), (18, "vertebrae T8"),
    (19, "vertebrae T9"), (20, "vertebrae T10"), (21, "vertebrae T11"), (22, "vertebrae T12"),
    (23, "vertebrae L1"), (24, "vertebrae L2"), (25, "vertebrae L3"), (26, "vertebrae L4"),
    (27, "vertebrae L5"),
    (28, "rib_cartilage"), (29, "sternum"),
    (30, "clavicles"), (31, "clavicle left"), (32, "clavicle right"),
    (33, "scapulas"), (34, "scapula left"), (35, "scapula right"),
    (36, "posterior 12th rib right"), (37, "posterior 12th rib left"),
    (38, "anterior 11th rib right"), (39, "posterior 11th rib right"),
    (40, "anterior 11th rib left"), (41, "posterior 11th rib left"),
    (42, "anterior 10th rib right"), (43, "posterior 10th rib right"),
    (44, "anterior 10th rib left"), (45, "posterior 10th rib left"),
    (46, "anterior 9th rib right"), (47, "posterior 9th rib right"),
    (48, "anterior 9th rib left"), (49, "posterior 9th rib left"),
    (50, "anterior 8th rib right"), (51, "posterior 8th rib right"),
    (52, "anterior 8th rib left"), (53, "posterior 8th rib left"),
    (54, "anterior 7th rib right"), (55, "posterior 7th rib right"),
    (56, "anterior 7th rib left"), (57, "posterior 7th rib left"),
    (58, "anterior 6th rib right"), (59, "posterior 6th rib right"),
    (60, "anterior 6th rib left"), (61, "posterior 6th rib left"),
    (62, "anterior 5th rib right"), (63, "posterior 5th rib right"),
    (64, "anterior 5th rib left"), (65, "posterior 5th rib left"),
    (66, "anterior 4th rib right"), (67, "posterior 4th rib right"),
    (68, "anterior 4th rib left"), (69, "posterior 4th rib left"),
    (70, "anterior 3rd rib right"), (71, "posterior 3rd rib right"),
    (72, "anterior 3rd rib left"), (73, "posterior 3rd rib left"),
    (74, "anterior 2nd rib right"), (75, "posterior 2nd rib right"),
    (76, "anterior 2nd rib left"), (77, "posterior 2nd rib left"),
    (78, "anterior 1st rib right"), (79, "posterior 1st rib right"),
    (80, "anterior 1st rib left"), (81, "posterior 1st rib left"),
    (82, "12th rib"),
    (83, "posterior 11th rib"), (84, "anterior 11th rib"),
    (85, "posterior 10th rib"), (86, "anterior 10th rib"),
    (87, "posterior 9th rib"), (88, "anterior 9th rib"),
    (89, "posterior 8th rib"), (90, "anterior 8th rib"),
    (91, "posterior 7th rib"), (92, "anterior 7th rib"),
    (93, "posterior 6th rib"), (94, "anterior 6th rib"),
    (95, "posterior 5th rib"), (96, "anterior 5th rib"),
    (97, "posterior 4th rib"), (98, "anterior 4th rib"),
    (99, "posterior 3rd rib"), (100, "anterior 3rd rib"),
    (101, "posterior 2nd rib"), (102, "anterior 2nd rib"),
    (103, "posterior 1st rib"), (104, "anterior 1st rib"),
    (105, "diaphragm"), (106, "left hemidiaphragm"), (107, "right hemidiaphragm"),
    (108, "stomach"), (109, "small bowel"), (110, "duodenum"), (111, "liver"),
    (112, "pancreas"), (113, "kidney left"), (114, "kidney right"),
    (115, "cardiomediastinum"), (116, "upper mediastinum"), (117, "lower mediastinum"),
    (118, "anterior mediastinum"), (119, "middle mediastinum"), (120, "posterior mediastinum"),
    (121, "heart"), (122, "heart atrium left"), (123, "heart atrium right"),
    (124, "heart myocardium"), (125, "heart ventricle left"), (126, "heart ventricle right"),
    (127, "aorta"), (128, "ascending aorta"), (129, "descending aorta"), (130, "aortic arch"),
    (131, "pulmonary artery"), (132, "pulmonary trunc"),
    (133, "left pulmonary artery "), (134, "right pulmonary artery "),
    (135, "inferior vena cava"), (136, "esophagus"),
    (137, "lung"), (138, "right lung"), (139, "left lung"),
    (140, "lung base"), (141, "mid zone lung"), (142, "upper zone lung"), (143, "apical zone lung"),
    (144, "right upper zone lung"), (145, "right mid zone lung"), (146, "right lung base"),
    (147, "right apical zone lung"), (148, "left upper zone lung"), (149, "left mid zone lung"),
    (150, "left lung base"), (151, "left apical zone lung"),
    (152, "lung lower lobe left"), (153, "lung upper lobe left"),
    (154, "lung lower lobe right"), (155, "lung middle lobe right"), (156, "lung upper lobe right"),
    (157, "less obstructed lung"), (158, "less obstructed right lung"),
    (159, "less obstructed left lung"), (160, "less obstructed lung base"),
    (161, "less obstructed mid zone lung"), (162, "less obstructed upper zone lung"),
    (163, "less obstructed apical zone lung"), (164, "less obstructed right upper zone lung"),
    (165, "less obstructed right mid zone lung"), (166, "less obstructed right lung base"),
    (167, "less obstructed right apical zone lung"), (168, "less obstructed left upper zone lung"),
    (169, "less obstructed left mid zone lung"), (170, "less obstructed left lung base"),
    (171, "less obstructed left apical zone lung"),
    (172, "trachea"), (173, "tracheal bifurcation"),
    (174, "breast"), (175, "breast left"), (176, "breast right"),
]


# =========================================================================== #
#  FRONTAL schema
# =========================================================================== #
_ORD = {1: "1st", 2: "2nd", 3: "3rd"}
# left CTR point is matched to the right one by rib number -> needs the posterior ribs
_CTR_VIA = ("right hemidiaphragm", "left hemidiaphragm", "right lung", "heart") + tuple(
    f"posterior {_ORD.get(n, f'{n}th')} rib {sd}" for n in range(4, 12) for sd in ("left", "right"))
_LMC_VIA = ("cardiomediastinum", "heart", "aortic arch", "pulmonary trunc", "upper mediastinum",
            "heart atrium left")
FRONTAL_RICH = {

    # ---- DIAPHRAGM ---------------------------------------------------------
    # NB: the "hemidiaphragm" mask is the FILLED sub-diaphragmatic (abdominal)
    # region down to the image bottom -- the diaphragm PROPER is its SUPERIOR
    # ridge (the dome curve). All diaphragm kpts are taken on that ridge
    # (ridge:*); never use centroid/area of this mask (= abdomen). The ridge is
    # sampled (medial -> dome -> lateral) so its curvature is captured
    # (flattening = hyperinflation / COPD).
    "left hemidiaphragm": ClassSchema(
        side="left",
        keypoints=[
            KP("dome", "dome_lateral", via=("heart",)),
            KP("ridge_lateral_third", "ridge:q_lateral"),
            KP("ridge_medial_third", "ridge:q_medial"),
            KP("costophrenic_diaphragmatic", "ridge:lateral",
               via=("left lung",), group="costophrenic_left"),
            KP("cardiophrenic_diaphragmatic", "ridge:medial",
               via=("heart",), group="cardiophrenic_left"),
        ],
        skeleton=[("cardiophrenic_diaphragmatic", "ridge_medial_third"),
                  ("ridge_medial_third", "dome"), ("dome", "ridge_lateral_third"),
                  ("ridge_lateral_third", "costophrenic_diaphragmatic")],
    ),
    "right hemidiaphragm": ClassSchema(
        side="right",
        keypoints=[
            KP("dome", "dome_lateral", via=("heart",)),
            KP("ridge_lateral_third", "ridge:q_lateral"),
            KP("ridge_medial_third", "ridge:q_medial"),
            KP("costophrenic_diaphragmatic", "ridge:lateral",
               via=("right lung",), group="costophrenic_right"),
            KP("cardiophrenic_diaphragmatic", "ridge:medial",
               via=("heart",), group="cardiophrenic_right"),
        ],
        skeleton=[("cardiophrenic_diaphragmatic", "ridge_medial_third"),
                  ("ridge_medial_third", "dome"), ("dome", "ridge_lateral_third"),
                  ("ridge_lateral_third", "costophrenic_diaphragmatic")],
    ),
    # combined frontal diaphragm mask is also the filled abdomen -> ridge only,
    # with per-side half-restriction (dome_left/right would otherwise coincide).
    "diaphragm": ClassSchema(
        side="both",
        keypoints=[
            KP("dome_left", "ridge:dome"),
            KP("dome_right", "ridge:dome"),
            KP("costophrenic_diaphragmatic_left", "ridge:lateral",
               group="costophrenic_left"),
            KP("costophrenic_diaphragmatic_right", "ridge:lateral",
               group="costophrenic_right"),
            KP("cardiophrenic_diaphragmatic_left", "ridge:medial",
               group="cardiophrenic_left"),
            KP("cardiophrenic_diaphragmatic_right", "ridge:medial",
               group="cardiophrenic_right"),
        ],
    ),

    # ---- LUNGS (whole) -----------------------------------------------------
    "left lung": ClassSchema(
        side="left",
        keypoints=[
            KP("apex", "extreme:top"),
            KP("lateral_margin", "lateral_at_hilum", via=("left pulmonary artery ", "heart", "cardiomediastinum", "tracheal bifurcation")),
            KP("costophrenic_pulmonary", "visible_corner:infero_lateral",
               via=("less obstructed left lung", "less obstructed right lung"), group="costophrenic_left"),
            # angle limbs (assistive): the costophrenic angle needs vertex + two limb points
            KP("costophrenic_limb_wall", "cp_limb:wall", via=("left hemidiaphragm", "right hemidiaphragm"),
               group="costophrenic_left"),
            KP("costophrenic_limb_diaphragm", "cp_limb:diaphragm",
               via=("less obstructed left lung", "less obstructed right lung"), group="costophrenic_left"),
            KP("cardiophrenic_pulmonary", "extreme:infero_medial",
               via=("heart",), group="cardiophrenic_left"),
            KP("hilum", "hilar_point", via=("left pulmonary artery ", "heart", "cardiomediastinum", "tracheal bifurcation")),
            # costal limb of the CP angle = infero-lateral-most rib-cage point
            KP("costophrenic_costal", "rib_costal",
               via=("posterior 8th rib left", "posterior 9th rib left",
                    "posterior 10th rib left"), group="costophrenic_left"),
            # CTR denominator endpoint (inner thorax at the dome level)
            KP("ctr_thorax", "index:ctr_thorax_rib", via=_CTR_VIA, group="ctr_thorax"),
            # mediastinal stripes (lung edge against a mediastinal structure);
            # NOT in the perimeter skeleton -- drawn as their own group line.
            KP("para_aortic_top", "contact:top", via=("descending aorta",), group="para_aortic"),
            KP("para_aortic_bottom", "contact:bottom", via=("descending aorta",), group="para_aortic"),
            KP("paraspinal_top", "contact:top", via=("spine",), group="paraspinal_left"),
            KP("paraspinal_bottom", "contact:bottom", via=("spine",), group="paraspinal_left"),
        ],
        skeleton=[("apex", "lateral_margin"), ("lateral_margin", "costophrenic_pulmonary"),
                  ("costophrenic_pulmonary", "cardiophrenic_pulmonary"),
                  ("cardiophrenic_pulmonary", "hilum"), ("hilum", "apex")],
    ),
    "right lung": ClassSchema(
        side="right",
        keypoints=[
            KP("apex", "extreme:top"),
            KP("lateral_margin", "lateral_at_hilum", via=("right pulmonary artery ", "heart", "cardiomediastinum", "tracheal bifurcation")),
            KP("costophrenic_pulmonary", "visible_corner:infero_lateral",
               via=("less obstructed left lung", "less obstructed right lung"), group="costophrenic_right"),
            # angle limbs (assistive): the costophrenic angle needs vertex + two limb points
            KP("costophrenic_limb_wall", "cp_limb:wall", via=("left hemidiaphragm", "right hemidiaphragm"),
               group="costophrenic_right"),
            KP("costophrenic_limb_diaphragm", "cp_limb:diaphragm",
               via=("less obstructed left lung", "less obstructed right lung"), group="costophrenic_right"),
            KP("cardiophrenic_pulmonary", "extreme:infero_medial",
               via=("heart",), group="cardiophrenic_right"),
            KP("hilum", "hilar_point", via=("right pulmonary artery ", "heart", "cardiomediastinum", "tracheal bifurcation")),
            KP("costophrenic_costal", "rib_costal",
               via=("posterior 8th rib right", "posterior 9th rib right",
                    "posterior 10th rib right"), group="costophrenic_right"),
            KP("ctr_thorax", "index:ctr_thorax", via=("right hemidiaphragm",), group="ctr_thorax"),
            KP("paraspinal_top", "contact:top", via=("spine",), group="paraspinal_right"),
            KP("paraspinal_bottom", "contact:bottom", via=("spine",), group="paraspinal_right"),
        ],
        skeleton=[("apex", "lateral_margin"), ("lateral_margin", "costophrenic_pulmonary"),
                  ("costophrenic_pulmonary", "cardiophrenic_pulmonary"),
                  ("cardiophrenic_pulmonary", "hilum"), ("hilum", "apex")],
    ),
    "lung": ClassSchema(
        side="both",
        keypoints=[
            KP("apex_left", "extreme:top"), KP("apex_right", "extreme:top"),
            KP("lateral_margin_left", "extreme:lateral"),
            KP("lateral_margin_right", "extreme:lateral"),
            KP("costophrenic_pulmonary_left", "extreme:infero_lateral",
               group="costophrenic_left"),
            KP("costophrenic_pulmonary_right", "extreme:infero_lateral",
               group="costophrenic_right"),
            KP("cardiophrenic_pulmonary_left", "extreme:infero_medial",
               group="cardiophrenic_left"),
            KP("cardiophrenic_pulmonary_right", "extreme:infero_medial",
               group="cardiophrenic_right"),
        ],
    ),

    # ---- LUNG LOBES (fissures = lobe<->lobe interface) ---------------------
    "lung upper lobe right": ClassSchema(side="right", keypoints=[
        KP("center", "centroid"),
        KP("minor_fissure", "contact:mid", via=("lung middle lobe right",),
           group="minor_fissure_right"),
        KP("oblique_fissure", "contact:mid", via=("lung lower lobe right",),
           group="oblique_fissure_right")]),
    "lung middle lobe right": ClassSchema(side="right", keypoints=[
        KP("center", "centroid"),
        KP("minor_fissure", "ridge_mid", via=("lung upper lobe right",),
           group="minor_fissure_right"),
        KP("oblique_fissure", "contact:mid", via=("lung lower lobe right",),
           group="oblique_fissure_right")]),
    "lung lower lobe right": ClassSchema(side="right", keypoints=[
        KP("center", "centroid"),
        KP("oblique_fissure", "contact:mid", via=("lung middle lobe right",),
           group="oblique_fissure_right")]),
    "lung upper lobe left": ClassSchema(side="left", keypoints=[
        KP("center", "centroid"),
        KP("oblique_fissure", "contact:mid", via=("lung lower lobe left",),
           group="oblique_fissure_left")]),
    "lung lower lobe left": ClassSchema(side="left", keypoints=[
        KP("center", "centroid"),
        KP("oblique_fissure", "contact:mid", via=("lung upper lobe left",),
           group="oblique_fissure_left")]),

    # ---- HEART + chambers --------------------------------------------------
    "heart": ClassSchema(
        side="both",
        keypoints=[
            KP("right_cardiac_border", "extreme:lateral"),    # CTR numerator (right)
            KP("left_cardiac_border", "extreme:lateral"),     # CTR numerator (left)
            KP("apex", "heart_apex", via=("left hemidiaphragm", "right hemidiaphragm")),  # LV apex above the left dome
            # vertex on the VISIBLE lung (heart mask continues below the diaphragm)
            KP("cardiophrenic_cardiac_left", "cph_limb:left:vertex",
               via=("less obstructed left lung", "less obstructed right lung", "left lung"),
               group="cardiophrenic_left"),
            KP("cardiophrenic_cardiac_right", "cph_limb:right:vertex",
               via=("less obstructed left lung", "less obstructed right lung", "right lung"),
               group="cardiophrenic_right"),
            KP("waist_cardiac", "lmc:waist", via=_LMC_VIA),
            # angle limbs (assistive) of the two cardiophrenic angles
            KP("cardiophrenic_left_limb_heart", "cph_limb:left:heart",
               via=("less obstructed left lung", "less obstructed right lung"), group="cardiophrenic_left"),
            KP("cardiophrenic_left_limb_diaphragm", "cph_limb:left:diaphragm",
               via=("less obstructed left lung", "less obstructed right lung"), group="cardiophrenic_left"),
            KP("cardiophrenic_right_limb_heart", "cph_limb:right:heart",
               via=("less obstructed left lung", "less obstructed right lung"), group="cardiophrenic_right"),
            KP("cardiophrenic_right_limb_diaphragm", "cph_limb:right:diaphragm",
               via=("less obstructed left lung", "less obstructed right lung"), group="cardiophrenic_right"),
        ],
        skeleton=[("waist_cardiac", "right_cardiac_border"),
                  ("right_cardiac_border", "cardiophrenic_cardiac_right"),
                  ("cardiophrenic_cardiac_right", "apex"),
                  ("apex", "cardiophrenic_cardiac_left"),
                  ("cardiophrenic_cardiac_left", "left_cardiac_border"),
                  ("left_cardiac_border", "waist_cardiac")],
    ),
    "heart atrium right": ClassSchema(
        side="right",
        keypoints=[KP("right_atrial_border", "extreme:lateral"),
                   KP("ivc_ra_atrial", "nearest", via=("inferior vena cava",), group="ivc_ra"),
                   KP("center", "centroid")],
    ),
    "heart atrium left": ClassSchema(
        side="left",
        keypoints=[KP("la_appendage", "lmc:laa", via=_LMC_VIA),
                   KP("double_density", "anat:right"),
                   KP("center", "centroid")],
    ),
    "heart ventricle left": ClassSchema(
        side="left", keypoints=[KP("apex", "extreme:bottom"), KP("center", "centroid")]),
    "heart ventricle right": ClassSchema(side="right", keypoints=[KP("center", "centroid")]),
    "heart myocardium": ClassSchema(side="both", keypoints=[KP("center", "centroid")]),

    # ---- GREAT VESSELS -----------------------------------------------------
    # tubular vessels carry a CENTERLINE (cl_0..cl_n along the medial axis) for
    # course/tortuosity, plus their salient landmark (knob, etc.).
    "aorta": ClassSchema(side="both", keypoints=[
        KP("knob", "extreme:lateral"),
        KP("cl_0", "centerline:0.0"), KP("cl_1", "centerline:0.25"),
        KP("cl_2", "centerline:0.5"), KP("cl_3", "centerline:0.75"),
        KP("cl_4", "centerline:1.0")],
        skeleton=[("cl_0", "cl_1"), ("cl_1", "cl_2"), ("cl_2", "cl_3"), ("cl_3", "cl_4")]),
    "ascending aorta": ClassSchema(
        side="right", keypoints=[KP("cl_0", "centerline:0.0"), KP("cl_1", "centerline:0.5"),
                                 KP("cl_2", "centerline:1.0")],
        skeleton=[("cl_0", "cl_1"), ("cl_1", "cl_2")]),
    "aortic arch": ClassSchema(
        side="left",
        keypoints=[KP("knob", "knob_silhouette", via=_LMC_VIA), KP("summit", "summit_distal"),
                   KP("ap_window_aortic", "lmc:ap_window", via=_LMC_VIA, group="ap_window")],
        skeleton=[("summit", "knob"), ("knob", "ap_window_aortic")],
    ),
    "descending aorta": ClassSchema(
        side="left",
        keypoints=[KP("cl_0", "centerline:0.0"), KP("cl_1", "centerline:0.33"),
                   KP("cl_2", "centerline:0.66"), KP("cl_3", "centerline:1.0")],
        skeleton=[("cl_0", "cl_1"), ("cl_1", "cl_2"), ("cl_2", "cl_3")],
    ),
    "pulmonary trunc": ClassSchema(
        side="both",
        keypoints=[KP("pa_segment", "lmc:pa_segment", via=_LMC_VIA),
                   KP("ap_window_pulmonic", "nearest", via=("aortic arch",), group="ap_window"),
                   KP("pa_bifurcation", "nearest",
                      via=("left pulmonary artery ", "right pulmonary artery "))],
    ),
    "pulmonary artery": ClassSchema(side="both", keypoints=[KP("center", "centroid")]),
    "left pulmonary artery ": ClassSchema(side="left", keypoints=[KP("hilum", "centroid")]),
    "right pulmonary artery ": ClassSchema(side="right", keypoints=[KP("hilum", "centroid")]),
    "inferior vena cava": ClassSchema(
        side="right",
        keypoints=[KP("ivc_ra_caval", "centerline:0.0", via=("heart atrium right",), group="ivc_ra"),
                   KP("bottom", "extreme:bottom")],
    ),
    "esophagus": ClassSchema(
        side="midline",
        keypoints=[KP("inlet", "centerline:0.0"), KP("cl_1", "centerline:0.4"),
                   KP("cl_2", "centerline:0.7"),
                   KP("gej_esophageal", "centerline:1.0", via=("stomach",), group="gej")],
        skeleton=[("inlet", "cl_1"), ("cl_1", "cl_2"), ("cl_2", "gej_esophageal")],
    ),

    # ---- AIRWAY ------------------------------------------------------------
    "trachea": ClassSchema(
        side="midline",
        keypoints=[KP("inlet", "centerline:0.0"), KP("cl_1", "centerline:0.5"),
                   KP("carina_tracheal", "nearest", via=("tracheal bifurcation",), group="carina"),
                   KP("clavicular_level", "trachea_at_clavicles", via=("clavicle left", "clavicle right")),
                   KP("right_tracheobronchial_angle", "tb_angle:right", via=("tracheal bifurcation",)),
                   # assistive: tracheal axis (deviation vs. the spinal axis from the vertebral corners)
                   KP("axis_upper", "trachea_axis:upper", via=("tracheal bifurcation",)),
                   KP("axis_lower", "trachea_axis:lower", via=("tracheal bifurcation",))],
        skeleton=[("inlet", "cl_1"), ("cl_1", "carina_tracheal")],
    ),
    # the bifurcation mask Y-splays on frontal -> carina apex + the two main-
    # bronchus tips give the carinal angle (angle at carina between the limbs).
    "tracheal bifurcation": ClassSchema(
        side="midline",
        keypoints=[KP("carina", "carina_notch", group="carina"),
                   KP("left_main_bronchus", "bronchus_mid:left"),
                   KP("right_main_bronchus", "bronchus_mid:right"),
                   # angle limbs (assistive) of the subcarinal angle, along the inferior bronchial walls
                   KP("subcarinal_limb_left", "subcarinal_limb:left", group="carina"),
                   KP("subcarinal_limb_right", "subcarinal_limb:right", group="carina")],
        skeleton=[("left_main_bronchus", "carina"), ("carina", "right_main_bronchus")],
    ),

    # ---- SHOULDER GIRDLE / STERNUM ----------------------------------------
    # SC = the clavicle's MEDIAL end (near midline), AC = its LATERAL end. Use
    # the bone's own extremes (robust to per-image L/R flips), not nearest-to-
    # partner (which mispairs when clavicle/scapula sit on opposite image sides).
    "clavicle left": ClassSchema(
        side="left",
        keypoints=[KP("sternoclavicular_clavicular", "extreme:medial", via=("sternum",),
                      group="sternoclavicular_left"),
                   KP("mid", "skel:mid"),
                   KP("acromioclavicular_clavicular", "extreme:lateral", via=("scapula left",),
                      group="acromioclavicular_left")],
        skeleton=[("sternoclavicular_clavicular", "mid"), ("mid", "acromioclavicular_clavicular")],
    ),
    "clavicle right": ClassSchema(
        side="right",
        keypoints=[KP("sternoclavicular_clavicular", "extreme:medial", via=("sternum",),
                      group="sternoclavicular_right"),
                   KP("mid", "skel:mid"),
                   KP("acromioclavicular_clavicular", "extreme:lateral", via=("scapula right",),
                      group="acromioclavicular_right")],
        skeleton=[("sternoclavicular_clavicular", "mid"), ("mid", "acromioclavicular_clavicular")],
    ),
    "clavicles": ClassSchema(
        side="both",
        keypoints=[KP("sternoclavicular_clavicular_left", "extreme:medial",
                      group="sternoclavicular_left"),
                   KP("acromioclavicular_clavicular_left", "extreme:lateral",
                      group="acromioclavicular_left"),
                   KP("sternoclavicular_clavicular_right", "extreme:medial",
                      group="sternoclavicular_right"),
                   KP("acromioclavicular_clavicular_right", "extreme:lateral",
                      group="acromioclavicular_right")],
    ),
    "sternum": ClassSchema(
        side="midline",
        keypoints=[KP("jugular_notch", "extreme:top"), KP("sternal_angle", "curv:centerline"),
                   KP("xiphoid", "extreme:bottom"),
                   KP("sternoclavicular_sternal_left", "nearest", via=("clavicle left",),
                      group="sternoclavicular_left"),
                   KP("sternoclavicular_sternal_right", "nearest", via=("clavicle right",),
                      group="sternoclavicular_right")],
        skeleton=[("jugular_notch", "sternal_angle"), ("sternal_angle", "xiphoid")],
    ),

    # ---- ABDOMINAL ---------------------------------------------------------
    "stomach": ClassSchema(side="left",
                           keypoints=[KP("gastric_fundus_apex", "fundus_under_diaphragm",
                                         via=("left hemidiaphragm", "right hemidiaphragm", "esophagus")),
                                      KP("gej_gastric", "nearest", via=("esophagus",), group="gej"),
                                      KP("center", "centroid")]),
    "liver": ClassSchema(side="right", keypoints=[KP("dome", "extreme:top"),
                                                  KP("center", "centroid")]),

    # ---- BREAST ------------------------------------------------------------
    "breast left": ClassSchema(side="left", keypoints=[KP("center", "centroid"),
                                                       KP("nipple", "todo")]),
    "breast right": ClassSchema(side="right", keypoints=[KP("center", "centroid"),
                                                         KP("nipple", "todo")]),
    "breast": ClassSchema(side="both", keypoints=[KP("center_left", "centroid"),
                                                  KP("center_right", "centroid")]),
}


def _side_of(name: str) -> str:
    if "left" in name:
        return "left"
    if "right" in name:
        return "right"
    return "both"


def _ordinal(num: str) -> str:
    n = int(num)
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


def _rib_number(name: str) -> str:
    for tok in name.split():
        if tok[:-2].isdigit() and tok[-2:] in ("st", "nd", "rd", "th"):
            return tok[:-2]
    return "?"


def _vertebra_schema_frontal(name: str) -> ClassSchema:
    # transverse processes = lateral tips of the FULL mask (outside the body);
    # the spinous process projects en face over the body on frontal -> not
    # separable here (it IS recoverable on lateral, see _vertebra_schema_lateral).
    kps = [KP("superior_left", "endplate:SL"), KP("superior_right", "endplate:SR"),
           KP("inferior_left", "endplate:IL"), KP("inferior_right", "endplate:IR"),
           KP("center", "body_centroid"),
           KP("transverse_process_left", "extreme:lateral"),
           KP("transverse_process_right", "extreme:lateral")]
    skel = [("superior_left", "superior_right"), ("superior_right", "inferior_right"),
            ("inferior_right", "inferior_left"), ("inferior_left", "superior_left")]
    level = name.replace("vertebrae ", "")
    if level.startswith("T"):
        num = level[1:]
        kps += [KP("costovertebral_vertebral_left", "nearest",
                   via=(f"posterior {_ordinal(num)} rib left",), group=f"costovertebral_{num}_left"),
                KP("costovertebral_vertebral_right", "nearest",
                   via=(f"posterior {_ordinal(num)} rib right",), group=f"costovertebral_{num}_right")]
    return ClassSchema(side="midline", keypoints=kps, skeleton=skel)


def _spine_segment_frontal() -> ClassSchema:
    return ClassSchema(side="midline",
                       keypoints=[KP("top", "skel:top"), KP("mid", "skel:mid"),
                                  KP("bottom", "skel:bottom")],
                       skeleton=[("top", "mid"), ("mid", "bottom")])


def _rib_schema_frontal(name: str) -> ClassSchema:
    side = _side_of(name)
    num = _rib_number(name)
    grp_side = side if side in ("left", "right") else "both"
    if "posterior" in name:
        kps = [KP("rib_head", "extreme:medial", group=f"costovertebral_{num}_{grp_side}"),
               KP("angle", "curv:centerline"),
               KP("anterior_tip", "extreme:lateral"),
               # assistive: 3 cm along the rib from the head -> rib slope (hyperinflation)
               KP("rib_slope_limb", "rib_slope_limb")]
    else:
        kps = [KP("costochondral_costal", "nearest", via=("rib_cartilage",),
                  group=f"costochondral_{num}_{grp_side}"),
               KP("mid", "skel:mid"),
               KP("anterior_end", "extreme:medial")]
    return ClassSchema(side=side, keypoints=kps,
                       skeleton=[(kps[0].name, kps[1].name), (kps[1].name, kps[2].name)])


def _lung_region_frontal(name: str) -> ClassSchema:
    side = _side_of(name)
    kps = [KP("center", "centroid")]
    if "apical" in name:
        kps.append(KP("apex", "plateau:top"))
    if "base" in name:
        partner = "left hemidiaphragm" if side == "left" else "right hemidiaphragm"
        grp = f"costophrenic_{side}" if side in ("left", "right") else "costophrenic"
        kps.append(KP("costophrenic_pulmonary", "extreme:infero_lateral",
                      via=(partner,), group=grp))
    return ClassSchema(side=side, keypoints=kps)


def _mediastinum_frontal(name: str) -> ClassSchema:
    kps = [KP("border_left", "extreme:lateral"), KP("border_right", "extreme:lateral"),
           KP("center", "centroid")]
    if name == "upper mediastinum":
        # right mediastinal-width endpoint (SVC edge) at the aortic-knob level
        kps.append(KP("mw_right", "mw_right", via=("aortic arch",) + _LMC_VIA))
    return ClassSchema(side="both", keypoints=kps)


def build_schema() -> dict:
    schema = {}
    for _id, name in CATEGORIES:
        if name in FRONTAL_RICH:
            schema[name] = FRONTAL_RICH[name]
        elif name in ("spine", "cervical spine", "thoracic spine", "lumbar spine"):
            schema[name] = _spine_segment_frontal()
        elif name.startswith("vertebrae "):
            schema[name] = _vertebra_schema_frontal(name)
        elif " rib" in name and name != "rib_cartilage":
            schema[name] = _rib_schema_frontal(name)
        elif name == "rib_cartilage":
            schema[name] = ClassSchema(side="both",
                                       keypoints=[KP("superior", "extreme:top"),
                                                  KP("inferior", "extreme:bottom"),
                                                  KP("center", "centroid")])
        elif "mediastinum" in name:
            schema[name] = _mediastinum_frontal(name)
        elif "lobe" in name:
            schema[name] = ClassSchema(side=_side_of(name), keypoints=[KP("center", "centroid")])
        elif "zone" in name or name.endswith("lung") or "lung base" in name or "lung" in name:
            schema[name] = _lung_region_frontal(name)
        elif name in ("scapula left", "scapula right"):
            schema[name] = ClassSchema(side=_side_of(name),
                                       keypoints=[KP("center", "centroid"),
                                                  KP("inferior_angle", "extreme:bottom"),
                                                  KP("glenoid", "extreme:supero_lateral")])
        elif name == "scapulas":
            schema[name] = ClassSchema(side="both", keypoints=[KP("center_left", "centroid"),
                                                               KP("center_right", "centroid")])
        else:
            schema[name] = ClassSchema(side=_side_of(name), keypoints=[KP("center", "centroid")])
    return schema


# =========================================================================== #
#  LATERAL schema  (anterior = toward sternum, posterior = toward spine)
# =========================================================================== #
LATERAL_RICH = {
    # both hemidiaphragms superimpose -> the combined "diaphragm" class.
    # Posterior costophrenic sulcus is the deepest recess (most sensitive for
    # effusion on the lateral film) -- the key lateral landmark.
    # combined diaphragm mask is likewise the filled abdomen -> ridge only.
    "diaphragm": ClassSchema(
        side="midline",
        keypoints=[KP("dome", "ridge:dome"),
                   KP("anterior_costophrenic", "sulcus_vertex:anterior",
                      via=("less obstructed lung", "lung"), group="anterior_costophrenic"),
                   KP("posterior_costophrenic", "sulcus_vertex:posterior",
                      via=("less obstructed lung", "lung"), group="posterior_costophrenic"),
                   # angle limbs (assistive) of both sulci
                   KP("anterior_costophrenic_limb_wall", "sulcus_limb:anterior:wall",
                      via=("less obstructed lung",), group="anterior_costophrenic"),
                   KP("anterior_costophrenic_limb_diaphragm", "sulcus_limb:anterior:diaphragm",
                      via=("less obstructed lung",), group="anterior_costophrenic"),
                   KP("posterior_costophrenic_limb_wall", "sulcus_limb:posterior:wall",
                      via=("less obstructed lung",), group="posterior_costophrenic"),
                   KP("posterior_costophrenic_limb_diaphragm", "sulcus_limb:posterior:diaphragm",
                      via=("less obstructed lung",), group="posterior_costophrenic"),
                   # assistive: sternodiaphragmatic angle (hyperinflation) from sternum + diaphragm
                   KP("sternodiaphragmatic_vertex", "sternodiaphragmatic:vertex", via=("sternum", "less obstructed lung"),
                      group="sternodiaphragmatic"),
                   KP("sternodiaphragmatic_limb_sternum", "sternodiaphragmatic:sternum", via=("sternum", "less obstructed lung"),
                      group="sternodiaphragmatic"),
                   KP("sternodiaphragmatic_limb_diaphragm", "sternodiaphragmatic:diaphragm", via=("sternum", "less obstructed lung"),
                      group="sternodiaphragmatic")],
        skeleton=[("anterior_costophrenic", "dome"), ("dome", "posterior_costophrenic")],
    ),
    "heart": ClassSchema(
        side="midline",
        keypoints=[KP("anterior_border", "extreme:anterior", via=("sternum",)),  # retrosternal
                   KP("posterior_border", "extreme:posterior", via=("spine",)),  # retrocardiac
                   KP("inferior", "extreme:bottom"),
                   # view extension: antero-inferior tip of the heart above the diaphragm
                   KP("apex", "apex_lateral", via=("diaphragm",)),
                   # resident addition (Delphi round 1): LV enlargement on the lateral film
                   KP("hoffman_rigler", "hoffman_rigler", via=("inferior vena cava",))],
        skeleton=[("anterior_border", "inferior"), ("posterior_border", "inferior")],
    ),
    "sternum": ClassSchema(  # fully meaningful on lateral (it is the anterior wall)
        side="midline",
        keypoints=[KP("jugular_notch", "extreme:antero_superior"),
                   KP("sternal_angle", "curv:centerline"),     # angle of Louis (~carina / T4-5)
                   KP("xiphoid", "extreme:antero_inferior"),
                   # assistive: manubriosternal angle (vertex = sternal angle)
                   KP("manubriosternal_limb_manubrium", "manubriosternal_limb:manubrium"),
                   KP("manubriosternal_limb_body", "manubriosternal_limb:body")],
        skeleton=[("jugular_notch", "sternal_angle"), ("sternal_angle", "xiphoid")],
    ),
    "trachea": ClassSchema(
        side="midline",
        keypoints=[KP("inlet", "centerline:0.0"), KP("cl_1", "centerline:0.5"),
                   KP("carina_tracheal", "nearest", via=("tracheal bifurcation",), group="carina")],
        skeleton=[("inlet", "cl_1"), ("cl_1", "carina_tracheal")],
    ),
    "tracheal bifurcation": ClassSchema(side="midline",
                                        keypoints=[KP("carina", "centroid", group="carina")]),
    "esophagus": ClassSchema(  # retrocardiac course on lateral
        side="midline",
        keypoints=[KP("inlet", "centerline:0.0"), KP("cl_1", "centerline:0.4"),
                   KP("cl_2", "centerline:0.7"),
                   KP("gej_esophageal", "centerline:1.0", via=("stomach",), group="gej")],
        skeleton=[("inlet", "cl_1"), ("cl_1", "cl_2"), ("cl_2", "gej_esophageal")],
    ),
    "aorta": ClassSchema(side="midline", keypoints=[          # arch seen in profile
        KP("cl_0", "centerline:0.0"), KP("cl_1", "centerline:0.25"),
        KP("cl_2", "centerline:0.5"), KP("cl_3", "centerline:0.75"),
        KP("cl_4", "centerline:1.0")],
        skeleton=[("cl_0", "cl_1"), ("cl_1", "cl_2"), ("cl_2", "cl_3"), ("cl_3", "cl_4")]),
    "aortic arch": ClassSchema(
        side="midline",
        keypoints=[KP("summit", "plateau:top"),       # middle of the arch's top (not a vessel origin)
                   KP("anterior", "extreme:anterior"), KP("posterior", "extreme:posterior")],
        skeleton=[("anterior", "summit"), ("summit", "posterior")],
    ),
    "ascending aorta": ClassSchema(side="midline",
                                   keypoints=[KP("retrosternal_posterior", "retrosternal:aortic",
                                                 via=("sternum",), group="retrosternal"),
                                              KP("cl_0", "centerline:0.0"),
                                              KP("cl_1", "centerline:0.5"),
                                              KP("cl_2", "centerline:1.0")],
                                   skeleton=[("cl_0", "cl_1"), ("cl_1", "cl_2")]),
    "descending aorta": ClassSchema(
        side="midline",
        keypoints=[KP("cl_0", "centerline:0.0"), KP("cl_1", "centerline:0.33"),
                   KP("cl_2", "centerline:0.66"), KP("cl_3", "centerline:1.0")],
        skeleton=[("cl_0", "cl_1"), ("cl_1", "cl_2"), ("cl_2", "cl_3")],
    ),
    "inferior vena cava": ClassSchema(          # view extension of the frontal IVC-RA junction
        side="midline", keypoints=[KP("ivc_ra_caval", "centerline:0.0", group="ivc_ra")]),
    "stomach": ClassSchema(side="midline", keypoints=[KP("gastric_fundus_apex", "extreme:top"),
                                                      KP("center", "centroid")]),
    "liver": ClassSchema(side="midline", keypoints=[KP("dome", "extreme:top"),
                                                    KP("center", "centroid")]),
}


def _vertebra_schema_lateral(name: str) -> ClassSchema:
    # profile view -> 4 body corners on the anterior-posterior / sup-inf axes;
    # anterior wedging here = compression fracture; sequence = kyphosis (Cobb).
    # spinous process = posterior-most tip of the mask (sticks out behind the
    # body on lateral; on frontal it is en face over the body -> not separable).
    kps = [KP("antero_superior", "endplate:AS"), KP("antero_inferior", "endplate:AI"),
           KP("postero_superior", "endplate:PS"), KP("postero_inferior", "endplate:PI"),
           KP("center", "body_centroid"),
           KP("spinous_process", "extreme:posterior")]
    skel = [("antero_superior", "postero_superior"), ("postero_superior", "postero_inferior"),
            ("postero_inferior", "antero_inferior"), ("antero_inferior", "antero_superior"),
            ("postero_superior", "spinous_process")]
    return ClassSchema(side="midline", keypoints=kps, skeleton=skel)


def _spine_segment_lateral() -> ClassSchema:
    return ClassSchema(side="midline",
                       keypoints=[KP("top", "skel:top"), KP("mid", "skel:mid"),
                                  KP("bottom", "skel:bottom")],
                       skeleton=[("top", "mid"), ("mid", "bottom")])


def _rib_schema_lateral(name: str) -> ClassSchema:
    # L/R superimpose; posterior ribs sweep down-and-back, anterior ribs forward.
    if "posterior" in name:
        kps = [KP("head", "nearest", via=("spine",)),   # costovertebral, near spine
               KP("angle", "curv:centerline"),
               KP("inferior_end", "extreme:postero_inferior")]
    else:
        kps = [KP("costochondral", "nearest", via=("rib_cartilage",)),
               KP("mid", "skel:mid"),
               KP("anterior_end", "extreme:anterior")]
    return ClassSchema(side="midline", keypoints=kps,
                       skeleton=[(kps[0].name, kps[1].name), (kps[1].name, kps[2].name)])


def _mediastinum_lateral(name: str) -> ClassSchema:
    # anterior / middle / posterior mediastinum are THE classic lateral
    # compartments -> anterior & posterior borders are the meaningful points.
    kps = [KP("anterior_border", "extreme:anterior"), KP("posterior_border", "extreme:posterior"),
           KP("center", "centroid")]
    if name == "anterior mediastinum":
        # retrosternal space: posterior sternal surface 3 cm below the sternal angle
        kps.append(KP("retrosternal_anterior", "retrosternal:sternal", via=("sternum", "ascending aorta"),
                      group="retrosternal"))
    return ClassSchema(side="midline", keypoints=kps)


def build_lateral_schema() -> dict:
    # only classes that actually appear on lateral images (see probe); the rest
    # get a centroid so the schema is total, but they will usually be v=0.
    schema = {}
    for _id, name in CATEGORIES:
        if name in LATERAL_RICH:
            schema[name] = LATERAL_RICH[name]
        elif name in ("spine", "cervical spine", "thoracic spine", "lumbar spine"):
            schema[name] = _spine_segment_lateral()
        elif name.startswith("vertebrae "):
            schema[name] = _vertebra_schema_lateral(name)
        elif " rib" in name and name != "rib_cartilage":
            schema[name] = _rib_schema_lateral(name)
        elif "mediastinum" in name:
            schema[name] = _mediastinum_lateral(name)
        elif "zone" in name or name.endswith("lung") or "lung base" in name or "lung" in name:
            kps = [KP("center", "centroid")]
            if "apical" in name:              # view extension of the frontal lung apex
                kps.append(KP("apex", "plateau:top"))
            if "base" in name:
                kps.append(KP("posterior_costophrenic", "extreme:postero_inferior",
                              via=("spine",), group="posterior_costophrenic"))
            schema[name] = ClassSchema(side="midline", keypoints=kps)
        else:
            schema[name] = ClassSchema(side="midline", keypoints=[KP("center", "centroid")])
    return schema


SCHEMA_FRONTAL = build_schema()
SCHEMA_LATERAL = build_lateral_schema()


if __name__ == "__main__":
    for label, s in (("FRONTAL", SCHEMA_FRONTAL), ("LATERAL", SCHEMA_LATERAL)):
        nk = sum(len(v.keypoints) for v in s.values())
        nn = sum(1 for v in s.values() for k in v.keypoints if k.method == "nearest")
        nt = sum(1 for v in s.values() for k in v.keypoints if k.method == "todo")
        print(f"{label}: {len(s)} classes, {nk} keypoints, {nn} nearest, {nt} todo")

    print("\n--- 'costophrenic_left' decomposed (FRONTAL) ---")
    for cname, v in SCHEMA_FRONTAL.items():
        for k in v.keypoints:
            if k.group == "costophrenic_left":
                print(f"  {cname:16s} -> {k.name:32s} {k.method}")

    print("\n--- LATERAL examples ---")
    for name in ("diaphragm", "heart", "sternum", "vertebrae T7", "posterior 7th rib",
                 "anterior mediastinum"):
        v = SCHEMA_LATERAL[name]
        print(f"[{name}] side={v.side}")
        for k in v.keypoints:
            via = f"  <- {', '.join(k.via)}" if k.via else ""
            print(f"    {k.name:24s} {k.method:26s}{via}")


# --------------------------------------------------------------------------- #
# Final vocabulary (Delphi round 1) -- prune the full rule set to the landmarks
# kept by the expert ratings. The full rule set stays available as *_FULL.
# --------------------------------------------------------------------------- #
import json as _json
import os as _os

SCHEMA_FRONTAL_FULL, SCHEMA_LATERAL_FULL = SCHEMA_FRONTAL, SCHEMA_LATERAL
FINAL_VOCAB_PATH = _os.environ.get(
    "KP_FINAL_VOCAB",
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "vocab", "final_vocab_v1.json"))


def _prune(schema: dict, keep: dict) -> dict:
    out = {}
    for cname, cls in schema.items():
        names = keep.get(cname, [])
        missing = [n for n in names if n not in cls.names()]
        if missing:
            raise KeyError(f"final vocab names without a rule: {cname}: {missing}")
        kps = [k for k in cls.keypoints if k.name in names]
        sk = [(a, b) for a, b in cls.skeleton if a in names and b in names]
        out[cname] = ClassSchema(side=cls.side, keypoints=kps, skeleton=sk)
    return out


if FINAL_VOCAB_PATH and _os.path.exists(FINAL_VOCAB_PATH):
    _v = _json.load(open(FINAL_VOCAB_PATH))["vocab"]
    SCHEMA_FRONTAL = _prune(SCHEMA_FRONTAL_FULL, _v["frontal"])
    SCHEMA_LATERAL = _prune(SCHEMA_LATERAL_FULL, _v["lateral"])
