from gnews import GNews
import sys
from datetime import datetime, timedelta

''''''
class GoogleNews(GNews):
    def __init__(self, *args, **kwargs):
        super(GoogleNews, self).__init__(*args, **kwargs)
  
    def get_news(self, text, start_date=None, end_date=None):
        start = ''
        end = ''

        if start_date != None:
            start=' after:{start} '.format(start=self.format_date_string(start_date))
        if end_date != None:
            end=' before:{end} '.format(end=self.format_date_string(end_date))

        search = text + start + end
        #print(search)
        return super().get_news(search)

    def format_date_string(self, date):
        date_str = ''
        if type(date) == tuple:
            try:
                if len(date) == 1:
                    date_str = date[0]
                else:
                    date_str = '{}-{}-{}'.format(date[0], date[1], date[2])
            except:
                print("Date tuple not formatted correctly. (YYYY) or (YYYY, MM, DD)", file=sys.stderr)
                assert(0)
        elif type(date) == datetime:
            date_str = date.__str__().split()[0]
        elif type(date) != str:
            print("Date must be a tuple, datetime or a string.", file=sys.stderr)
            assert(0)
        else:
            date_str = date

        return date_str

'''
Args: List of lists [A, B, date] or singleton list and GoogleNews.
Search topics A and B on date (date - 1 : date)
Return: (newsA, newsB) where newsA is a list of items
returned by get_news : [get_news(A, date - 1, date)]

Expect date as 'yyyy-mm-dd'
'''
def getnews(queries, gnews):
    newsA = []
    newsB = []

    def process_query(query):
        A = query[0]
        B = query[1]
        date = query[2]
        date = datetime.fromisoformat(date)
        
        td = timedelta(days=1)
        end = date - td
        start = end - td

        newsA.append(gnews.get_news(A, start, end))
        newsB.append(gnews.get_news(B, start, end))

    if queries.ndim == 1:
        process_query(queries)
    else:
        for query in queries:
            process_query(query)

    return newsA, newsB




    
'''
Args: List of queries followed by GoogleNews object
query: [(text, [(fr, to)])]
    Where text is the string to search,
            fr are the start dates,
            to are the end dates
        fr and to may be None

Returns: List of tuples of a list: [(text, [results])]

        Where each index in the outer list corresponds to the 
        original query, and each index in the inner list is an actual article. 
        
        len of inner list can be decided by 'max_results=x'
        in arg for GoogleNews

def getnews(queries, gnews):
    return [ (text, [ gnews.get_news(text, date[0], date[1]) 
                                                    for date in dates ]) 
                    for text, dates in queries ]

'''
'''
Input:
[('Tottenham', [((2022, 11, 11), (2022, 11, 12))]), 
 ('Liverpool', [(None, None))],
 ('Brighton', [((2022, 11, 11), (2022, 11, 12)), ('2022-11-04', '2022-11-05')])]

 Output:
 [('Tottenham', [x articles]),
  ('Liverpool', [x articles]),
  ('Brighton', [x articles, x articles])]

'''

'''if __name__=='__main__':
    gnews=GoogleNews(language='en')
    print(gnews.get_news('Tottenham', (2022, 11, 10), (2022, 11, 11))[0])'''