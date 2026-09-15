from abc import ABC, abstractmethod
from itertools import combinations
import math

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
        pass

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
        pass


class OutDoorSeating(Feature):
    """Feature indiquant la présence ou l'absence de places assises en terrasse."""

    def check(self, poi):
        """Retourne la valeur du tag OSM `outdoor_seating` ("yes", "no", ou "None")."""
        if poi.get("outdoor_seating") is not None:
            return poi.get("outdoor_seating")
        else: return "None"

    def to_text(self, value):
        """
        Traduit la valeur `outdoor_seating` en fragment de phrase.

        Returns:
            "places extérieures", "pas de places extérieures", ou None si
            la valeur est inconnue.
        """
        if value=="yes":
            return "outdoor_seating"
        if value=="no":
            return " "
        return None


class InDoorSeating(Feature):
    """Feature indiquant la présence ou l'absence de places assises en salle."""

    def check(self, poi):
        """Retourne la valeur du tag OSM `indoor_seating` ("yes", "no", ou "None")."""
        if poi.get("indoor_seating") is not None:
            return poi.get("indoor_seating")
        else: return "None"

    def to_text(self, value):
        """
        Traduit la valeur `indoor_seating` en fragment de phrase.

        Returns:
            "places intérieures", "pas de places intérieures", ou None si
            la valeur est inconnue.
        """
        if value=="yes":
            return "indoor_seating"
        if value=="no":
            return " "
        return None


class CookingType(Feature):
    """Feature représentant le type de cuisine d'un établissement restauration."""

    def check(self, poi):
        """
        Retourne le type de cuisine depuis le tag OSM `cuisine`.

        Normalise les valeurs absentes sous toutes leurs formes (None, NaN float,
        chaîne "nan") en retournant None.
        """
        value = poi.get("cuisine")
        # Catch tous les cas : None, "nan", float NaN
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        if str(value).lower() == "nan":
            return None
        return value

    def to_text(self, value):
        """Retourne "cuisine <type>" si le type est renseigné, sinon None."""
        if value is not None and value!="nan":
            return f"cuisine {value}"


class CloseToSubway(Feature):
    """Feature listant les stations de métro situées à moins de 600 m du POI."""

    COLUMN = "close_to_subway"

    def check(self, poi):
        """
        Retourne la liste des noms de stations proches depuis la colonne pré-calculée.

        Raises:
            KeyError: Si la colonne `close_to_subway` est absente du GeoDataFrame,
                indiquant que `generate_proximity_feature` n'a pas encore été appelé.

        Returns:
            list[str] | None: Liste de noms de stations, ou None si aucune station
                n'est proche ou si la valeur est manquante.
        """
        if self.COLUMN not in poi.index:
            raise KeyError(f"La colonne '{self.COLUMN}' est absente du GeoDataFrame. Avez-vous calculé les distances aux stations ?")
        value = poi[self.COLUMN]
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        if str(value).lower() == "nan":
            return None
        return value

    def to_text(self, value):
        """
        Génère les fragments "proche de la station X" pour chaque station de la liste.

        Les valeurs float résiduelles (NaN non filtrés) sont ignorées silencieusement.
        Seules les stations dont le nom est une chaîne valide sont conservées.

        Returns:
            list[str] | None: Fragments de proximité, ou None si aucune station valide.
        """
        if not value:
            return None
        if isinstance(value, float):
            print(value)
            return
        if isinstance(value, list):
            value = [s for s in value if isinstance(s, str)]
        if not value:
            return None
        result = []
        for r in range(1, 2):
            for combo in combinations(value, r):
                if len(combo) == 1:
                    result.append(f"proche de la station {combo[0]}")
                else:
                    result.append(f"proche des stations {', '.join(combo[:-1])} et {combo[-1]}")
        return result

class CloseToParc(Feature):
    """Feature listant les parcs situés à moins de 600 m du POI."""

    COLUMN = "close_to_parc"

    def check(self, poi):
        """
        Retourne la liste des noms de parcs proches depuis la colonne pré-calculée.

        Raises:
            KeyError: Si la colonne `close_to_parc` est absente du GeoDataFrame,
                indiquant que `generate_proximity_feature` n'a pas encore été appelé.

        Returns:
            list[str] | None: Liste de noms de parcs, ou None si aucun parc
                n'est proche ou si la valeur est manquante.
        """
        if self.COLUMN not in poi.index:
            raise KeyError(f"La colonne '{self.COLUMN}' est absente du GeoDataFrame. Avez-vous calculé les distances aux parcs ?")
        value = poi[self.COLUMN]
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        if str(value).lower() == "nan":
            return None
        return value

    def to_text(self, value):
        """
        Génère les fragments "proche du parc X" pour chaque parc de la liste.

        Les valeurs float résiduelles (NaN non filtrés) sont ignorées silencieusement.
        Seuls les parcs dont le nom est une chaîne valide sont conservés.

        Returns:
            list[str] | None: Fragments de proximité, ou None si aucun parc valide.
        """
        if not value:
            return None
        if isinstance(value, float):
            print(value)
            return
        if isinstance(value, list):
            value = [s for s in value if isinstance(s, str)]
        if not value:
            return None
        result = []
        for r in range(1, 2):
            for combo in combinations(value, r):
                if len(combo) == 1:
                    result.append(f"proche du parc {combo[0]}")
                else:
                    result.append(f"proche des parcs {', '.join(combo[:-1])} et {combo[-1]}")
        return result