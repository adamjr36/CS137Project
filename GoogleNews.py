from gnews import GNews
import sys

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
        elif type(date) != str:
            print("Date must be a tuple or a string.", file=sys.stderr)
            assert(0)
        else:
            date_str = date
    
        return date_str

if __name__=='__main__':
    gnews=GoogleNews(language='en')
    print(gnews.get_news('Tottenham', (2022, 11, 10), (2022, 11, 11))[0])