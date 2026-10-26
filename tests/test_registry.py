"""Le registre des templates reste aligné sur le code.

Le projet tient **deux** registres : `registry.REGISTRY`, peuplé à l'import par
le décorateur `@template` de chaque module, et `schema.TEMPLATE_REGISTRY`, écrit
à la main parce qu'il porte en plus les colonnes de contexte et le libellé
attendu dans l'énoncé. Rien ne les synchronise, et ils sont indexés
différemment — par nom de fonction d'un côté, par nom court de l'autre — ce qui
rend toute divergence invisible à l'œil.

Ces tests vivaient dans `test_dataset_io.py` jusqu'à la suppression de la couche
de persistance, dont ils ne dépendaient pas.
"""

from benchmark_pipeline.generator.template_question.schema import TEMPLATES_BY_NAME

#: Modules de `geospatial/` qui ne sont pas des templates de question et n'ont donc
#: rien à faire dans le registre. `make_question_geo` est un combinateur encore
#: en chantier — il croise features et questions géo — et ne suit pas le contrat
#: `make_question_*(df, …, nb_q, seed) -> DataFrame`. Le recenser ici plutôt que
#: de relâcher le test garde la garantie : tout *autre* module ajouté à `geospatial/`
#: fera échouer `test_every_template_module_is_in_the_registry`.
NON_TEMPLATE_MODULES = {"make_question_geo"}


def test_registry_matches_the_modules_on_disk():
    """Chaque entrée du registre désigne un module et un générateur réels.

    Le registre pilote la paramétrisation de toute la suite : une entrée
    périmée ferait silencieusement disparaître un template de la couverture.
    """
    import importlib

    from benchmark_pipeline.generator.template_question.schema import TEMPLATE_REGISTRY
    from tests.conftest import template_source

    for template in TEMPLATE_REGISTRY:
        module_path = template_source(template)
        assert module_path.exists(), f"{module_path} est introuvable"
        assert module_path.stem == template.name, (
            f"{template.name} est déclaré dans {template.module}, dont le "
            f"fichier s'appelle {module_path.name}"
        )

        module = importlib.import_module(template.module)
        assert callable(getattr(module, template.generator, None)), (
            f"{template.generator} absent de {template.module}"
        )


def test_every_template_module_is_in_the_registry():
    """Aucun template sur disque n'échappe au registre."""
    from tests.conftest import geospatial_dir

    on_disk = {
        p.stem for p in geospatial_dir().glob("*.py")
        if not p.stem.startswith("_")
    } - NON_TEMPLATE_MODULES
    assert on_disk == set(TEMPLATES_BY_NAME), (
        f"registre et disque divergent : "
        f"seulement sur disque {sorted(on_disk - set(TEMPLATES_BY_NAME))}, "
        f"seulement au registre {sorted(set(TEMPLATES_BY_NAME) - on_disk)}"
    )


def test_the_two_registries_describe_the_same_templates():
    """Le registre par décorateur et celui du schéma ne doivent pas diverger.

    Un template décoré mais absent du schéma échappe à toute la suite ;
    l'inverse ferait échouer la paramétrisation. Ce test est le seul endroit
    qui les confronte.
    """
    import benchmark_pipeline.generator.template_question.geospatial  # noqa: F401 — peuple REGISTRY
    from benchmark_pipeline.generator.template_question.registry import REGISTRY

    decorated = set(REGISTRY)
    declared = {t.generator for t in TEMPLATES_BY_NAME.values()}

    assert decorated == declared, (
        f"les deux registres divergent : décorés mais absents du schéma "
        f"{sorted(decorated - declared)}, déclarés au schéma mais non décorés "
        f"{sorted(declared - decorated)}"
    )


def test_registry_generators_share_the_nb_q_and_seed_contract():
    """Tout générateur accepte `nb_q` et `seed` en mot-clé.

    C'est le contrat sur lequel reposent `conftest.run_template` et la boucle
    de `make_question_composite`, qui appellent un générateur sans savoir
    lequel c'est. Les signatures diffèrent par ailleurs — `list_direction`,
    `min_size`, `corridor_m`, `ratio_cat_q` — mais ces deux-là doivent rester
    communs, sinon le lot ne peut plus être produit d'un seul appel paramétré.
    """
    import importlib
    import inspect

    from benchmark_pipeline.generator.template_question.schema import TEMPLATE_REGISTRY

    for template in TEMPLATE_REGISTRY:
        module = importlib.import_module(template.module)
        params = inspect.signature(getattr(module, template.generator)).parameters
        missing = [p for p in ("nb_q", "seed") if p not in params]
        assert not missing, (
            f"{template.generator} n'accepte pas {missing} : le lot ne peut "
            f"pas être produit par un appel uniforme"
        )
