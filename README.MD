
### Tissue regeneration/wound healing or cancer metastasis

# Migration-Proliferation Dichotomy (MPD): Cell proliferation (division) and migration (movement)

## The Interplay (Migration vs. Proliferation)
Typically, a cell cannot strongly migrate and rapidly divide at the same time.The Dichotomy: To move, a cell must reorganize its cytoskeleton, which restricts the cell cycle. Conversely, to divide, a cell must round up and detach from surrounding structures, limiting its ability to crawl.In Wound Healing: At the leading edge of a wound, cells become hyper-motile to close the gap. Behind this leading edge is a proliferative hub where cells rapidly divide to replenish the cell population.In Cancer: In tumor progression, cells on the leading edge of the tumor prioritize migration to invade surrounding tissues, whereas the interior tumor cells focus on proliferation.



## Algorithm
The initial state of flow of fluid is generated using standard formula of 
# Kinetic energy of plasma molecules considering the biomarkers, viscosity and other factors of blood quality of the patient in initial state
plasma molecules then are segmented as they are are the formula is tried to embed to the tracking algorithm as initial state and then the individual flow cost function is generated
After the cost function, there is no need to segment for cell identification, the model can track the cell, with timely segmentation to correspond it with intial cell that is supposed to be there
The cells might be dividing, for which the life cycle of the cell is also to be taken under consideration for matching, of the identified cells, that are supposed to be in the position.


# after finding the ara with the most proabability of the object to be located is identified, then apply, method like 
## Tracking by Contour Evolution
# For tracking: establish temporal associations between the segmented cells
Can be done using <multiframe sliding window>


#  Cell Tracking Challenges (CTC I–III)


# 1.State representation: For each cell, maintain
centroid,
velocity estimate (derived from the previous two or three frames),
segmentation mask,
appearance features (or a learned embedding).
# 2.Motion gating: Use the velocity estimate only to define a search region around the predicted location. For example, if a cell has moved approximately (Δx, Δy) over recent frames, search within a radius centered at (x_t + Δx, y_t + Δy) rather than the entire image.

# 3.Final association: Compute the matching cost using multiple cues:
predicted position error,
mask overlap (IoU),
appearance similarity,
shape similarity.

# 4.Global optimization: Use the Hungarian algorithm or graph optimization to determine the final assignments.
