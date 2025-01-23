import json
import requests

class EnrichrAnalysis:

    legend: tuple[str] = ('Rank', 'Term name', 'P-value', 'Odds ratio', 'Combined score',
              'Overlapping genes', 'Adjusted p-value', 'Old p-value', 'Old adjusted p-value')

    def __init__(self, glist:list[str], caller: callable):
        self.glist=glist
        self.glist_id = None
        self.res = None

        self.caller = caller(self)

    def analyze(self):
        self.caller()

class EnrichrCaller:

    url: str = 'https://maayanlab.cloud/Enrichr/'
    db: str = "KEGG_2015"

    def __init__(self, container: "EnrichrAnalysis"):
        self.cont = container

    def add_list(self, desc: str="NA"):
        q = self.url+'addList'
        payload = dict(list=(None,"\n".join(self.cont.glist)),
                       description=(None, desc))

        response = requests.post(q, files=payload)
        if not response.ok:
            raise Exception('Error analyzing gene list')
        self.cont.glist_id = json.loads(response.text)["userListId"]


    def enrich(self):
        q = self.url + f'enrich?userListId={self.cont.glist_id}&backgroundType={self.db}'
        response = requests.get(q)
        if not response.ok:
            raise Exception('Error fetching enrichment results')

        self.cont.res = json.loads(response.text)

    def __call__(self, *args, **kwargs):
        '''
        Send a list of genes to Enrichr to get information about enriched pathways
        >> glist = ["ENTPD5","FBXO3","TRAP1", "KDR", "SEPHS2", "EXOSC4", "RILP","HOXA7"]
        >> enran=EnrichrAnalysis(glist, EnrichrCaller)
        >> enran.analyze()
        >> print(enran.res)
        Args:
            *args:
            **kwargs:

        Returns:

        '''
        self.add_list()
        self.enrich()
        print("Done with Enrichr analysis")


def main():
    pass