import logging
import re

logger = logging.getLogger(__name__)


class ContainsEntity:
    def __init__(self):
        self.orgs = re.compile(
            "(D(?:e(?:fense (?:C(?:o(?:unterintelligence|mmissary|ntract)|riminal)|T(?:echn(?:ology|ical)|hreat)|In(?:telligence|formation)|A(?:cquisition|dvanced)|H(?:UMINT|ealth|uman)|P(?:risoner|OW/MIA)|L(?:ogistics|egal)|Security|Finance|Nuclear|Media)|p(?:uty (?:Secretary|Under)|artment of))|irector(?:, (?:(?:Strategy|Command),|(?:Intelligen|For)ce|Operation(?:al|s)|Logistics|Manpower|Joint)| of)|oD Components)|C(?:o(?:m(?:p(?:onent [Hh]eads|uter Emergency)|batant Command(?:er)?s|mandant of)|nstruction Engineering|ld Regions)|h(?:ief (?:(?:Informatio|Huma)n|Financial)|airman, Joint)|entral (?:Intelligence|Security)|(?:ybersecurity|ustoms) and|lose Combat)|N(?:a(?:tional (?:(?:(?:Reconnaissa|Intellige)nc|Defens)e|G(?:eospatial-Intelligence|uard)|A(?:eronautics|ssessment)|Security)|val (?:Criminal|Research))|orth(?: (?:Atlant|Pacif)ic|western Division))|U(?:\\.S\\. (?:(?:Transportatio|Europea|Norther)n|S(?:trategic|outhern|pecial)|C(?:entral|yber)|A(?:frica|rmy)|Indo-Pacific)|n(?:i(?:f(?:ormed Services|ied Combatant)|ted States)|der Secretary))|A(?:rm(?:y (?:(?:Nation|Medic)al|Re(?:search|view)|Digitization|Corps|and)|ed (?:Servi|For)ces)|ir (?:Education|Mobility|National|Combat|Force)|ssistant (?:Commandant|Secretary|to))|M(?:i(?:ss(?:i(?:ssippi Valley|le Defense)|ouri River)|litary (?:Academy,|Postal))|a(?:rine (?:(?:Force|Corp)s|Expeditionary)|jor Commands))|S(?:e(?:lective Service|rgeant Major|cretary of)|outh(?: (?:Atlant|Pacif)ic|western Division)|pace (?:Development|and)|upreme Court)|V(?:eterans (?:Employment|Benefits|Health|Day)|ice (?:Chairman|Director),)|G(?:eneral (?:Services|Counsel)|overnment Accountability|reat Lakes)|Joint (?:P(?:ersonnel|rogram)|(?:Chief|Force)s|Artificial|History)|E(?:lectromagnetic Spectrum|xecutive Secretary|mployer Support)|W(?:a(?:shington Headquarters|terways Experiment)|hite Sands)|P(?:acific (?:Ocean|Air)|rotecting Critical|entagon Force)|F(?:ederal Voting|leet Forces)|O(?:rganization|ffice) of|Topographic Engineering)"  # noqa
        )
        self.abbrvs = re.compile(
            r"(D(?:e(?:fense (?:C(?:o(?:unterintelligence|mmissary|ntract)|riminal)|T(?:echn(?:ology|ical)|hreat)|In(?:telligence|formation)|A(?:cquisition|dvanced)|H(?:UMINT|ealth|uman)|P(?:risoner|OW/MIA)|L(?:ogistics|egal)|Security|Finance|Nuclear|Media)|p(?:uty (?:Secretary|Under)|artment of))|irector(?:, (?:(?:Intelligen|For)ce|C(?:ommand,|entral)|Operation(?:al|s)|Logistics|Strategy,|Manpower|National|Joint)| of)|oD Components)|C(?:o(?:m(?:mand(?:er, (?:(?:Theate|Ai)r|U(?:nited|.S.)|Detainee|Special|North|joint)|ant of)|p(?:onent [Hh]eads|uter Emergency)|batant Command(?:er)?s)|nstruction Engineering|ld Regions)|h(?:ief (?:(?:Informatio|Huma)n|Financial)|airman, Joint)|entral (?:Intelligence|Security)|(?:ybersecurity|ustoms) and|lose Combat)|N(?:a(?:tional (?:(?:(?:Reconnaissa|Intellige)nc|Defens)e|G(?:eospatial-Intelligence|uard)|A(?:eronautics|ssessment)|Security)|val (?:Criminal|Research))|orth(?: (?:Atlant|Pacif)ic|western Division))|A(?:rm(?:y (?:(?:Nation|Medic)al|Re(?:search|view)|Digitization|Corps|and)|ed (?:Servi|For)ces)|ir (?:Education|Mobility|National|Combat|Force)|ssistant (?:(?:Secretar|Deput)y|Commandant|to))|U(?:.S. (?:(?:Transportatio|Europea|Norther)n|S(?:trategic|outhern|pecial)|C(?:entral|yber)|A(?:frica|rmy)|Indo-Pacific)|n(?:i(?:f(?:ormed Services|ied Combatant)|ted States)|der Secretary))|M(?:i(?:ss(?:i(?:ssippi Valley|le Defense)|ouri River)|litary (?:Academy,|Postal))|a(?:rine (?:(?:Force|Corp)s|Expeditionary)|jor Commands))|S(?:e(?:lective Service|rgeant Major|cretary of)|outh(?: (?:Atlant|Pacif)ic|western Division)|pace (?:Development|and)|upreme Court)|V(?:eterans (?:Employment|Benefits|Health|Day)|ice (?:Chairman|Director),)|G(?:eneral (?:Services|Counsel)|overnment Accountability|reat Lakes)|Joint (?:P(?:ersonnel|rogram)|(?:Chief|Force)s|Artificial|History)|E(?:lectromagnetic Spectrum|xecutive Secretary|mployer Support)|P(?:acific (?:Ocean|Air)|rotecting Critical|entagon Force)|F(?:ederal Voting|leet Forces)|O(?:rganization|ffice) of|Topographic Engineering|Washington Headquarters)"
        )

    def __call__(self, text):
        """
        Simple check if it matches either regular expressions. Return `True` if
        either regular expressions match.

        Args:
            text (str): the text

        Returns:
            Bool
        """
        if self._re_search(text, self.abbrvs):
            return True
        elif self._re_search(text, self.orgs):
            return True
        else:
            return False

    @staticmethod
    def _re_search(text, regex):
        mobj = re.search(regex, text)
        if mobj is None:
            return False
        elif len(mobj.group(1)) > 1:
            return True
        else:
            return False
