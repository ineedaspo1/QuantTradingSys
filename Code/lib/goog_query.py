"""MakMan Google Scrapper & Mass Exploiter
 
Usage:
  scrap.py <search> <pages>
  scrap.py (-h | --help)
 
Arguments:
  <search>  String to be Searched
  <pages>   Number of pages
 
Options:
  -h, --help     Show this screen.
 
"""
 
import  requests, re
from    docopt import docopt
from    bs4 import BeautifulSoup
from    time import time as timer
 
 
def get_urls(search_string, start):
    #Empty temp List to store the Urls
    temp        = []
    url         = 'http://www.google.com/search'
    payload     = { 'q' : search_string, 'start' : start }
    my_headers  = { 'User-agent' : 'Mozilla/11.0' }
    r           = requests.get( url, params = payload, headers = my_headers )
    soup        = BeautifulSoup( r.text, 'html.parser' )
    h3tags      = soup.find_all( 'h3', class_='r' )
    for h3 in h3tags:
        try:
            temp.append( re.search('url\?q=(.+?)\&sa', h3.a['href']).group(1) )
        except:
            continue
    return temp
 
def main():
    start     = timer()
    #Empty List to store the Urls
    result    = []
    arguments = docopt( __doc__, version='MakMan Google Scrapper & Mass Exploiter' )
    search    = "journalism"
    pages     = 1
    #Calling the function [pages] times.
    for page in range( 0,  int(pages) ):
        #Getting the URLs in the list
        result.extend( get_urls( search, str(page*10) ) )
    #Removing Duplicate URLs
    result    = list( set( result ) )
    print( *result, sep = '\n' )
    print( '\nTotal URLs Scraped : %s ' % str( len( result ) ) ) 
    print( 'Script Execution Time : %s ' % ( timer() - start, ) )
 
if __name__ == '__main__':
    main()
 
#End