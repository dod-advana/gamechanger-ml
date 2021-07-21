import logging
import re

logger = logging.getLogger(__name__)


class ContainsEntity:
    def __init__(self):
        self.orgs = re.compile(
            "(D(?:e(?:fense (?:C(?:o(?:unterintelligence|mmissary|ntract)|riminal)|T(?:echn(?:ology|ical)|hreat)|In(?:telligence|formation)|A(?:cquisition|dvanced)|H(?:UMINT|ealth|uman)|P(?:risoner|OW/MIA)|L(?:ogistics|egal)|Security|Finance|Nuclear|Media)|p(?:uty (?:Secretary|Under)|artment of))|irector(?:, (?:(?:Intelligen|For)ce|C(?:ommand,|entral)|Operation(?:al|s)|Logistics|Strategy,|Manpower|National|Joint)| of)|oD Components)|C(?:o(?:m(?:mand(?:er, (?:(?:Theate|Ai)r|U(?:nited|.S.)|Detainee|Special|North|joint)|ant of)|p(?:onent [Hh]eads|uter Emergency)|batant Command(?:er)?s)|nstruction Engineering|ld Regions)|h(?:ief (?:(?:Informatio|Huma)n|Financial)|airman, Joint)|entral (?:Intelligence|Security)|(?:ybersecurity|ustoms) and|lose Combat)|N(?:a(?:tional (?:(?:(?:Reconnaissa|Intellige)nc|Defens)e|G(?:eospatial-Intelligence|uard)|A(?:eronautics|ssessment)|Security)|val (?:Criminal|Research))|orth(?: (?:Atlant|Pacif)ic|western Division))|A(?:rm(?:y (?:(?:Nation|Medic)al|Re(?:search|view)|Digitization|Corps|and)|ed (?:Servi|For)ces)|ir (?:Education|Mobility|National|Combat|Force)|ssistant (?:(?:Secretar|Deput)y|Commandant|to))|U(?:.S. (?:(?:Transportatio|Europea|Norther)n|S(?:trategic|outhern|pecial)|C(?:entral|yber)|A(?:frica|rmy)|Indo-Pacific)|n(?:i(?:f(?:ormed Services|ied Combatant)|ted States)|der Secretary))|M(?:i(?:ss(?:i(?:ssippi Valley|le Defense)|ouri River)|litary (?:Academy,|Postal))|a(?:rine (?:(?:Force|Corp)s|Expeditionary)|jor Commands))|S(?:e(?:lective Service|rgeant Major|cretary of)|outh(?: (?:Atlant|Pacif)ic|western Division)|pace (?:Development|and)|upreme Court)|V(?:eterans (?:Employment|Benefits|Health|Day)|ice (?:Chairman|Director),)|G(?:eneral (?:Services|Counsel)|overnment Accountability|reat Lakes)|Joint (?:P(?:ersonnel|rogram)|(?:Chief|Force)s|Artificial|History)|E(?:lectromagnetic Spectrum|xecutive Secretary|mployer Support)|P(?:acific (?:Ocean|Air)|rotecting Critical|entagon Force)|F(?:ederal Voting|leet Forces)|O(?:rganization|ffice) of|Topographic Engineering|Washington Headquarters)"
        )
        self.abbrvs = re.compile(
            r"(C(?:D(?:R(?:US(?:(?:S(?:O(?:UTH)?|PACE|TRAT)|C(?:YBER|ENT)|INDOPA|NORTH|TRANS)COM|A(?:FRICOM|RNORTH)|E(?:LEMNORAD|UCOM))|(?:JSO|SOJ)TF|AFNORTH|NORAD|TSOC)|O)|E(?:N(?:WD(?: (?:MR|NP))?|TCOM|AD)|(?:S[APW]|LR|MV|PO)D|C(?:ER|RL)|TEC|WES|RT)|I(?:[CO]|S?A)|(?:/C)?FO|M[CO]|CLTF|HCOC|APE|BP|SS)|D(?:I(?:R(?:(?:SPACE|MOB)FOR|NSA)|S?A)|O(?:D(?: (?:TRMC|EA))?|T\&E)|(?:/CI|LS?|EC|M)A|e(?:pSecDef|CA)|C(?:[AMS]A|IS)|A(?:RPA|\&M|U)|T(?:[RS]A|IC)|H(?:R?A|M)|N(?:FSB|I)|P(?:AA|MO)|J(?: 7|S)|S(?:CA|S)|oD CIO|FAS)|A(?:SD(?:S(?:O/L(?:IC&)?IC)?|HD&(?:ASA|GS)|OEPP|NII|RA)?|F(?:R(?:[HL]|(?:OT)?C)|(?:S[OP]|GS|M)C|OSI)?|D(?:USDTP|O)|R(?:BA|NG|L)|TSD(?:IO|PA)|M(?:EDD|C)|(?:ET|C)C|&S|NG)|U(?:S(?:C(?:E(?:NTCOM|RT)|YBERCOM|G)|S(?:(?:O(?:UTH)?|TRAT)COM|F)|(?:(?:INDO)?PA|TRANS|EU)COM|M(?:C(?: CID)?|EPCOM|A)|A(?:F(?:RICOM|E)?|CE)?|N(?:ORTHCOM|A)|UHS|FF)|CC)|S(?:O(?:(?:UTH)?COM|/LIC\&IC)|(?:COTU|S)S|TRATCOM|ecDef|DA|MC)|N(?:A(?:SA|G)|C(?:IS|B)|ORTHCOM|G[AB]|R[LO]|[DI]U|avy|SA)|M(?:A(?:JCOM[Ss]|RFORCOM)|DA)|E(?:MSO CFT|xecSec|UCOM|SGR)|P(?:AC(?:AF|OM)|CTTF|FPA|\&R)|J(?:(?:AI|FS)C|PEOCBRND|CS)|O(?:N[IR]|USDC|CMO|EA)|V(?:[BH]A|DNC|ETS)|(?:H(?:D\%AS)?|R)A|I(?:P?SA|\&E)?|W(?:SMR|HS)|G(?:AO|SA)|L(?:\&MR|A)|TRANSCOM|FVAP)"
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
