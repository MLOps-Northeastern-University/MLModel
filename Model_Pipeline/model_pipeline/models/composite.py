"""Several models presented as one.

Owner: P2.

Each part owns some targets. The composite fits each part on its own targets'
labels with the same folds and seed, joins their predictions, and saves one
folder whose manifest lists the parts and their versions. Two parts can't own the
same target. To everything downstream it's an ordinary model.
"""
