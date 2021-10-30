from elasticsearch import Elasticsearch
from elasticsearch.serializer import JSONSerializer


class ElasticSearch:
    def __init__(self):
        self.host_ip_address = '10.90.90.202'
        self.host_port = 9200
        self.username = 'viewer'
        self.password = 'kDZUL93un5V5r5Qw'
        self.es = None

    def connect(self):
        if self.es is None:
            self.es = Elasticsearch(
                f'http://{self.username}:{self.password}@{self.host_ip_address}:{self.host_port}/'
            )
            return True
        else:
            return False

    def disconnect(self):
        if self.es is None:
            return False
        else:
            self.es.close()
            self.es = None
            return True

    def search(self):
        if self.es is None:
            return False
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"paperAbstract": "MRI"}},
                        {"match": {"paperAbstract": "cancer"}},
                        {"match": {"paperAbstract": "brain"}},
                        # {"match": {"paperAbstract": "analysis"}}
                    ],
                    "filter": [
                        {"match": {"fieldsOfStudy": "Computer Science"}}
                    ]
                }
            }
        }
        res = self.es.search(body=query,
                             index='papers',
                             _source=True,
                             size=10000)
        with open('papers.json', 'w', encoding='utf-8') as f:
            f.write(JSONSerializer().dumps(data=[hit['_source'] for hit in res['hits']['hits']]))
        total_count = self.es.count(query)['count']


