import urllib.request, urllib.parse, json

query = '''
SELECT ?item ?itemLabel ?image ?materialLabel ?collectionLabel ?date WHERE {
  { ?item wdt:P2596 wd:Q216788. }
  UNION
  { ?item wdt:P195 wd:Q6940790. }
  
  OPTIONAL { ?item wdt:P18 ?image. }
  OPTIONAL { ?item wdt:P186 ?material. }
  OPTIONAL { ?item wdt:P195 ?collection. }
  OPTIONAL { ?item wdt:P571 ?date. }
  
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
'''

url = 'https://query.wikidata.org/sparql?query=' + urllib.parse.quote(query)
req = urllib.request.Request(url, headers={'Accept': 'application/sparql-results+json', 'User-Agent': 'ChampaThesisBot/0.1'})
try:
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read())
        bindings = res.get('results', {}).get('bindings', [])
        print(f"Total results: {len(bindings)}")
        print(json.dumps(bindings[:5], indent=2))
except Exception as e:
    print('Error:', e)
