import math
from abc import ABC, abstractmethod


class Feature(ABC):
    """
    Classe de base abstraite pour toutes les features descriptives d'un POI.

    Une feature encapsule deux responsabilités : extraire une valeur brute depuis
    les attributs d'un POI (`check`), puis la traduire en fragment de langue
    naturelle utilisable dans une question de benchmark (`to_text`).
    Toute feature concrète doit implémenter ces deux méthodes.
    """

    @abstractmethod
    def check(self, poi) -> bool | float:
        """
        Extrait la valeur brute de la feature pour un POI donné.

        Args:
            poi (Series): Ligne d'un GeoDataFrame représentant un point d'intérêt.

        Returns:
            La valeur de la feature (str, bool, float, list, etc.), ou None si
            l'information est absente ou non applicable.
        """

    @abstractmethod
    def to_text(self, value) -> str:
        """
        Convertit la valeur brute en fragment de phrase en langue naturelle.

        Args:
            value: Valeur retournée par `check`.

        Returns:
            str | list[str] | None: Fragment(s) de phrase prêt(s) à être intégré(s)
                dans une question, ou None si la valeur ne produit pas de texte utile.
        """


class OutDoorSeating(Feature):
    """Feature indiquant la présence ou l'absence de places assises en terrasse."""
    COLUMN = "outdoor_seating"
    def check(self, poi):
        """Retourne la valeur du tag OSM `outdoor_seating` ("yes", "no", ou "None")."""
        if poi.get(self.COLUMN) is not None:
            return poi.get(self.COLUMN)
        else: return "None"

    def to_text(self, value):
        """
        Traduit la valeur `outdoor_seating` en fragment de phrase.

        Returns:
            "places extérieures", "pas de places extérieures", ou None si
            la valeur est inconnue.
        """
        if value=="yes":
            return "with outdoor seating"
        if value=="no":
            return "without outdoor seating"
        return None


class InDoorSeating(Feature):
    """Feature indiquant la présence ou l'absence de places assises en salle."""
    COLUMN = "indoor_seating"
    
    def check(self, poi):
        """Retourne la valeur du tag OSM `indoor_seating` ("yes", "no", ou "None")."""
        if poi.get(self.COLUMN) is not None:
            return poi.get(self.COLUMN)
        else: return "None"

    def to_text(self, value):
        """
        Traduit la valeur `indoor_seating` en fragment de phrase.

        Returns:
            "places intérieures", "pas de places intérieures", ou None si
            la valeur est inconnue.
        """
        if value=="yes":
            return "with indoor seating"
        if value=="no":
            return "without indoor seating"
        return None


class CookingType(Feature):
    """Feature représentant le type de cuisine d'un établissement restauration."""
    COLUMN = "cooking_type"
    
    def check(self, poi):
        """
        Retourne le type de cuisine depuis le tag OSM `cooking_type`.

        Normalise les valeurs absentes sous toutes leurs formes (None, NaN float,
        chaîne "nan") en retournant None.
        """
        value = poi.get(self.COLUMN)
        # Catch tous les cas : None, "nan", float NaN
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        if str(value).lower() == "nan":
            return None
        return value

    def to_text(self, value):
        """Retourne "cooking_type <type>" si le type est renseigné, sinon None."""
        if value is not None and value!="nan":
            return f"cooking_type {value}"


class Category(Feature):
    COLUMN = "category"

    def check(self, poi):
        value = poi.get(self.COLUMN)
        # Catch tous les cas : None, "nan", float NaN
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        if str(value).lower() == "nan":
            return None
        return value

    def to_text(self, value):
        """Retourne "category <type>" si la cat est renseignée, sinon None."""
        if value is not None and value!="nan":
            return f"category {value}"