import logging as log
import os

import requests

# StockTwits details
ST_BASE_URL = 'https://api.stocktwits.com/api/2/'
ST_BASE_PARAMS = dict(access_token=os.getenv('ST_ACCESS_TOKEN'))

# Example list of exchanges to limit a watchlist to
EXCHANGES = ['NYSE', 'NASDAQ', 'NYSEMkt', 'NYSEArca']


# ---------------------------------------------------------------------
# Basic StockTwits interface
# ---------------------------------------------------------------------
def get_watched_stocks(wl_id):
    """ Get list of symbols being watched by specified StockTwits watchlist
    """
    wl = requests.get_json(ST_BASE_URL + 'watchlists/show/{}.json'.format(wl_id), params=ST_BASE_PARAMS)
    wl = wl['watchlist']['symbols']
    return [s['symbol'] for s in wl]


def get_stock_stream(symbol):
    """ gets stream of messages for given symbol
    """
    all_params = ST_BASE_PARAMS.copy()
    for k, v in params.iteritems():
        all_params[k] = v
    return requests.get_json(ST_BASE_URL + 'streams/symbol/{}.json'.format(symbol), params=all_params)


def get_message_stream(wl_id, params={}):
    """ Gets up to 30 messages from Watchlist (wl_id) according to additional params
    """
    all_params = ST_BASE_PARAMS.copy()
    for k, v in params.iteritems():
        all_params[k] = v
    return requests.get_json(ST_BASE_URL + 'streams/watchlist/{}.json'.format(wl_id), params=all_params)


def add_to_watchlist(symbols, wl_id):
    """ Adds list of symbols to our StockTwits watchlist.  Returns list of new symbols added
    """
    deadline = 30 * len(symbols)
    symbols = ','.join(symbols)  # must be a csv list
    params = ST_BASE_PARAMS.copy()
    params['symbols'] = symbols
    resp = requests.post_json(ST_BASE_URL + 'watchlists/{}/symbols/create.json'.format(wl_id), params=params,
                              deadline=deadline)
    if resp['response']['status'] == 200:
        return [s['symbol'] for s in resp['symbols']]
    else:
        return []


def delete_from_watchlist(symbol, wl_id):
    """ removes a single "symbols" (str) from watchlist.  Returns True on success, False on failure
    """
    params = ST_BASE_PARAMS.copy()
    params['symbols'] = symbol
    resp = requests.post_json(ST_BASE_URL + 'watchlists/{}/symbols/destroy.json'.format(wl_id), params=params)
    if resp['response']['status'] == 200:
        return True
    else:
        return False


def get_trending_stocks():
    """ returns list of trending stock symbols, ensuring each symbol is part of a NYSE or NASDAQ
    """
    trending = requests.get_json(ST_BASE_URL + 'trending/symbols.json', params=ST_BASE_PARAMS)['symbols']
    symbols = [s['symbol'] for s in trending if s['exchange'] in EXCHANGES]
    return symbols


def clean_watchlist(wl_id):
    """ Deletes stocks to follow if they aren't part of NASDAQ or NYSE
    """
    wl = requests.get_json(ST_BASE_URL + 'watchlists/show/{}.json'.format(wl_id),
                           params=ST_BASE_PARAMS)['watchlist']['symbols']
    qty_deleted = 0
    for sym in wl:
        if sym['exchange'] not in EXCHANGES:
            log.info("Removing {}".format(sym))
            if delete_from_watchlist(sym['symbol'], wl_id=wl_id):
                qty_deleted += 1
            else:
                log.error("Error deleting symbol from watchlist: {}".format(sym))
    return qty_deleted


if __name__ == '__main__':
    symbols = ['AAPL', 'BLK', 'BA', 'BE', 'BABA']
    data = get_trending_stocks()
