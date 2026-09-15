"""Générateurs de questions de type A.

Importer ce package suffit à peupler `src.template_question.registry.REGISTRY` :
chaque module ci-dessous applique `@template` à sa fonction `make_question_*`,
et un décorateur ne s'exécute qu'à l'import de son module.
"""

from . import (
    area_border,
    area_direction,
    area_inside,
    area_outside,
    point_between,
    point_near,
    point_near_cardinal,
    point_near_metric,
    point_towards,
    street_along,
    street_cross,
    street_opposite_side,
)
