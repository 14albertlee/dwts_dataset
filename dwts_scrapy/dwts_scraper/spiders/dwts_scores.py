import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

from ..items import DwtsScraperItem

from scrapy.shell import inspect_response

import pandas as pd
import re
import janitor
import io
import sys
from nameparser import HumanName
import numpy as np

# Debug prints below (e.g. unmatched celebrity/professional names) can contain
# non-ASCII characters (accents, etc.). On Windows the console/log stream defaults
# to cp1252, which can't encode them and would otherwise crash the whole crawl.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")



class DwtsScoresSpider(CrawlSpider):
    name = 'dwts_scores'
    allowed_domains = ['en.wikipedia.org']
    start_urls = ['https://en.wikipedia.org/wiki/Dancing_with_the_Stars_(American_season_1)']

    rules = (
        # Wikipedia renamed season pages from "(American_season_N)" to
        # "(American_TV_series)_season_N" at some point after season 31; the old
        # URLs still redirect, but outgoing links on the pages use the new form.
        # Match both so the crawl keeps discovering newer seasons. The old-style
        # URLs are given higher scheduling priority (see _prioritize below)
        # because some seasons (e.g. 15) only have a correctly structured judge
        # panel paragraph on the old-style page; parse_item only processes the
        # first page seen for a given season, so old-style pages must win the
        # race deterministically rather than by luck of crawl order.
        Rule(LinkExtractor(allow=[r'wiki/Dancing_with_the_Stars_\(American_season_\d+\)$']),
             callback='parse_item', follow=True, process_request='_prioritize'),
        Rule(LinkExtractor(allow=[r'wiki/Dancing_with_the_Stars_\(American_TV_series\)_season_\d+$']),
             callback='parse_item', follow=True),
#       Rule(LinkExtractor(allow=r'wiki/Dancing_with_the_Stars_\(American_season_11\)$'), callback='parse_item', follow=True),
    )

    def _prioritize(self, request, response):
        # See the comment on `rules` above: bump old-style season URLs ahead of
        # everything else in the scheduling queue so they're parsed first.
        return request.replace(priority=request.priority + 1)

    def name_parse(self, str_name):
        h_n = HumanName(str_name)
        return (' '.join(getattr(h_n, comp) for comp in ['first', 'middle']))

    def parse_item(self, response):
        # inspect_response(response, self)
        
        # item = DwtsScraperItem()
        
        # Title is "... (American season N)" on older pages or
        # "... (American TV series) season N" on pages Wikipedia has since renamed.
        season = response.xpath('string(//*[@id="firstHeading"]/text())').re(r'(\d+)\)?$')[0]

        # Both the old "(American_season_N)" and new "(American_TV_series)_season_N"
        # URLs are crawled (old cached pages still link with the old form; live
        # pages link with the new form), so the same season can be reached twice
        # under different URLs. Parse each season only once.
        if not hasattr(self, 'seen_seasons'):
            self.seen_seasons = set()
        if season in self.seen_seasons:
            self.logger.info('Skipping duplicate fetch of season %s at %s', season, response.url)
            return
        self.seen_seasons.add(season)

        # Create Cast dictionary.
        cast_section = response.xpath('//*[@id="Couples"]') 
        cast_table = cast_section.xpath('../following-sibling::table[1]')
        
        cast_pandas = ( 
            pd.read_html(io.StringIO(cast_table.get().replace('<br>','---')))[0].fillna('') # read_html returns list of tables
              .rename(columns=lambda x: re.sub(r'\[.*?\]','',x)) # remove footnote refs from column headers.
              .clean_names()
        )
        
        # get pro lookup table.
 #       cast_pros = cast_pandas.filter(['professional_partner'])
 #       cast_pros['list_of_pros'] = cast_pros['professional_partner'].str.split('---')
        all_pros = (
            cast_pandas
            .filter(['professional_partner'])
            .assign(list_of_pros = lambda df_: df_.professional_partner.str.split('---'))
            .filter(['list_of_pros'])
            .explode(['list_of_pros'])
            .rename(columns={'list_of_pros': 'professional'})
            .assign(professional = lambda df_: df_.professional.str.replace(r'\(Week.*?\)', '', regex = True).str.strip(' '))
            .drop_duplicates()
            .assign(pro_name_part = lambda df_: df_.professional
                                                   .apply(self.name_parse)
                                                   .str.strip()
                                                   .replace(['Maksim'], 'Maks')
                                                   .replace(['Valentin'], 'Val')
                   ) 
        )
        
        # In one season there is both Anna Demidova ('Anna D.' and Anna Trebunskaya 'Anna T.'
        # https://en.wikipedia.org/wiki/Dancing_with_the_Stars_(American_season_9)
        if (all_pros.professional.str.contains("Anna Demidova").any()):
            # df_final['judge'] = np.where(df_final['judge'] == 'Guest judge', df_final['guest_judge'], df_final['judge'])
            all_pros.pro_name_part = np.where(all_pros.professional == "Anna Demidova", "Anna D.", all_pros.pro_name_part)
            all_pros.pro_name_part = np.where(all_pros.professional == "Anna Trebunskaya", "Anna T.", all_pros.pro_name_part)
            
        
        all_celebs = (
            cast_pandas
            .rename(columns={'notability_known_for_': 'notability'})
            .filter(['celebrity','notability'])
            .assign(
                celebrity = lambda df_: df_.celebrity.str.replace(r'---S.*?$', '', regex = True), 
                celeb_name_part = lambda df_: df_.celebrity
                                                     # .replace(['Mike "The Miz" Mizanin'], 'Miz Mizanin')
                                                     # .replace(['Bill Engvall'], 'Bill E. Engvall')
                                                     # .replace(['Bill Nye'], 'Bill N. Nye')
                                                      # removes ---Season 7 all-stars
                                                     # .str.replace(r'---S.*?$', '', regex = True) 
                                                     .apply(self.name_parse)
                                                     .str.strip()
                                                     .replace(['Brian Austin'], 'Brian')
                                                     .replace(['Miz'], 'The Miz')
                                                     .replace(['Jessie James'], 'Jessie')
                                                     .replace(['Metta World'], 'Metta')
                                                     .replace(['Apolo Anton'], 'Apolo')
                                                     .replace(['P'], 'Master P')
                                                     .replace(['Melissa Joan'], 'Melissa')
                                                     .replace(['Lil\''], 'Lil\' Kim')  
                                                     .replace(['David Alan'], 'David')
                                                     .replace(['Marissa Jaret'], 'Marissa')
                                                     .replace(['Vivica A.'], 'Vivica')
                                                     .replace(['Candace Cameron'], 'Candace')
                                                     # .replace(['D.L'], 'D.L.') # fixed on page
                                                     .replace(['The'], 'The Situation') # hope there is only one
                                                     .replace(['Vanilla'], 'Vanilla Ice')
                                                     .replace(['Jake T.'], 'Jake')
                                                     .replace(['Jennie Finch'], 'Jennie')
                                                     .replace(['Elizabeth Berkley'], 'Elizabeth')
                                                     
                   )
        )
        # Note the above can be replaced by putting all the special cases into a dict (much more natural)
        # and using .replace(to_replace = name_special_cases).
        
        if (all_celebs.celebrity.str.contains("Bill Nye").any()):
            all_celebs.celeb_name_part = np.where(all_celebs.celebrity == "Bill Nye", "Bill N.",  all_celebs.celeb_name_part)
            all_celebs.celeb_name_part = np.where(all_celebs.celebrity == "Bill Engvall", "Bill E.",  all_celebs.celeb_name_part)
        
        all_celebs.celeb_name_part = np.where(all_celebs.celebrity == 'Mike "The Miz" Mizanin', 'The Miz',  all_celebs.celeb_name_part)
        all_celebs.celeb_name_part = np.where(all_celebs.celebrity == 'Sailor Brinkley-Cook', 'Sailor',  all_celebs.celeb_name_part)
        all_celebs.celeb_name_part = np.where(all_celebs.celebrity == 'Mr. T', 'Mr. T',  all_celebs.celeb_name_part)
        
        # Note that these can likely be done with Series.where
        
        weekly_scores = response.xpath('//*[@id="Weekly_scores"]')
        # print(weekly_scores.get())
        
        # Currently this gets all of the judges on the page.  Season 18 somehow has GMA as a judge?
        # Ah, it's a link somewhere in one of these judge sentences.
        # Probably need to get the first preceeding sibling matching for each table
        
       # if (default_judge_sentences := weekly_scores.xpath('../following-sibling::p[1]')):
       #     default_judge_html = default_judge_sentences[0] # gets only first judge statement
       #     judge_sentence = ''.join(default_judge_html.xpath('./descendant-or-self::*/text()').getall())
        did_find_guest = False
        # print(judge_sentence)
        
        # tags = weekly_scores.xpath('../following-sibling::*')
        # for tag in tags:
        #     print(tag)
        
        # To parse this we need to iterate through the descendants in document order;
        # older Wikipedia pages have a flat h2/h3/table/p structure, but pages have since
        # moved to wrapping each heading/week in nested <section> elements. Using the
        # following:: axis (rather than following-sibling::) flattens either structure
        # back into the same linear sequence this loop expects.
        
        tags = weekly_scores.xpath('following::*[self::h2 or self::h3 or self::table or self::p]')

        # Old pages have the judge-panel paragraph as a following-sibling of the heading;
        # new pages nest it inside a <section>, so following-sibling::p finds nothing.
        # The real judge-panel sentence is the intro paragraph right after the "Weekly
        # scores" heading, before the first week's <h3> (any <p> between an <h3> and its
        # table is a week-specific note, not the judge panel, e.g. season 15's "Couples
        # performed either the cha-cha-cha or foxtrot..."). So take the first <p> seen
        # before the first <h3>; if a page has none there, fall back to not having one.
        default_judge_sentence = ''
        for tag in tags:
            tag_name = tag.xpath('name()').get()
            if tag_name == "h3":
                break
            elif tag_name == "p":
                default_judge_sentence = ''.join(tag.xpath('./descendant-or-self::*/text()').getall())
                break

        # As of late 2025, Wikipedia's season 15 article no longer has a
        # judge-panel sentence anywhere on the page (it was removed in a
        # past edit), so the scan above always comes back empty for every
        # week. Season 15 used a single fixed judging panel all season
        # (no guest judges), in the same left-to-right order as the
        # per-judge score columns reconstructed above, so hardcode it
        # rather than silently losing all of season 15's per-judge scores.
        if not default_judge_sentence and season == '15':
            default_judge_sentence = 'Carrie Ann Inaba, Len Goodman, Bruno Tonioli'

        for index, tag in enumerate(tags):
           # print(tag.get())
            tag_name = tag.xpath('name()').get()
            if tag_name == "h2":  # Moves out of Weekly Scores
                break
            
            elif tag_name == "h3": # new week
                # Older pages hold the heading text in a child <span class="mw-headline">;
                # newer pages put it directly in the <h3>. string(.) covers both.
                week_title = tag.xpath('string(.)').get()
           #     week = re.search(r'Week (\d+)').group(1)
           #     week_theme = week_title.split(": ")[-1]
                
            elif tag_name == "table": # new table, check for judge statement
                # reset either way so not carried over to next table
                did_find_guest = False
                
                # judge statement usually comes just before the table
                # but sometimes it can be a few p up.
                # https://en.wikipedia.org/w/index.php?title=Dancing_with_the_Stars_(American_season_11)#Week_7:_200th_Episode_Week
                # https://en.wikipedia.org/w/index.php?title=Dancing_with_the_Stars_(American_season_22)#Week_4:_Disney_Night
                # check preceeding-sibling until hit either <table> or <h3>
                
                # trouble in https://en.wikipedia.org/wiki/Dancing_with_the_Stars_(American_season_31)#Week_6:_Michael_Bubl%C3%A9_Night
                # tags[39]
                
              #  print(tag.get()) 
            
                # Look backwards in the tag list using index.  weird results with preceding-sibling
                keep_looking = True
                earlier_index = index
                while (keep_looking):
                    earlier_index -= 1
                    previous_tag = tags[earlier_index]
    
                    if previous_tag.xpath('name()').get() in ["table","h3","h2"]:
                        if not did_find_guest:
                            judge_sentence = default_judge_sentence
                        break # stop searching previous tags above table.
                        
                    elif previous_tag.xpath('name()').get() == "p":
                        p_text = ''.join(previous_tag.xpath('./descendant-or-self::*/text()').getall())
                        if re.search(r'Individual judge', p_text):
                            did_find_guest = True
                            keep_looking = False
                            judge_sentence = p_text
                        
                raw_table = (
                    pd.read_html(io.StringIO(tag.get().replace('<br>','---')))[0].fillna('') # read_html returns list of tables
                        # remove footnote refs from column headers.
                    .rename(columns=lambda x: re.sub(r'\[.*?\]','',x))
                    .clean_names()
                )

                # A few seasons' tables (e.g. season 15) break each judge's score into
                # its own column (one column per judge surname under a shared "scores"
                # header) instead of a single combined "scores" column, which produces
                # a MultiIndex here. Reconstruct the usual "total (j1, j2, j3)" string
                # so the rest of this parser (which expects a flat "scores" column)
                # can handle these tables the same way as all the others.
                if isinstance(raw_table.columns, pd.MultiIndex):
                    judge_cols = [c for c in raw_table.columns if c[0] == 'scores']
                    judge_col_names = ['_'.join(c) for c in judge_cols]
                    # .rename(columns=...) doesn't flatten a MultiIndex (it tries to
                    # rename per-level), so replace .columns outright instead.
                    raw_table.columns = [(c[0] if c[0] == c[1] else '_'.join(c)) for c in raw_table.columns]
                    if judge_cols and 'total_score' in raw_table.columns:
                        raw_table['scores'] = (
                            raw_table['total_score'].astype(str) + ' ('
                            + raw_table[judge_col_names].astype(str).agg(', '.join, axis=1)
                            + ')'
                        )
                    else:
                        self.logger.warning(
                            'Skipping table with unsupported column layout in %s %s: %s',
                            season, week_title, raw_table.columns.tolist(),
                        )
                        continue

                if 'couple' not in raw_table.columns:
                    self.logger.warning(
                        'Skipping table with unsupported column layout in %s %s: %s',
                        season, week_title, raw_table.columns.tolist(),
                    )
                    continue

                score_pandas = (
                    raw_table
                    .assign(season = season,
                            week_title = week_title,
                            judge_phrase = judge_sentence.split(': ')[-1],
                            couple = lambda df_: df_.couple.str.replace(r'\[\w\]','',regex = True),
                           )
                    .assign(couple = lambda df_: df_.couple.replace(['D.L & Cheryl'],'D.L. & Cheryl'))
                    .query('~couple.str.contains("---")') # removes multiple couple dances.  revisit this.
                    .query('~couple.isnull()')
                    # Multi-couple (e.g. team) dances are sometimes rendered without "---"
                    # between couples (depends on how the Wikipedia table cell used <br>
                    # vs. other separators), leaving a single "A & B C & D" string that
                    # doesn't split into exactly celeb/pro. Drop those too, consistent
                    # with the "---" based exclusion above.
                    .loc[lambda df_: df_.couple.str.count(r'\s&\s') == 1]
                )
                
                # Note that the split might be better with:
                # (age ... .str.split('-', expand=True) ... .iloc[:,0] ... .astype(int)
                
                if(score_pandas.shape[0] < 1):
                    continue
                
                score_pandas[['celeb_name_part', 'pro_name_part']] = score_pandas['couple'].str.split(" & ", expand = True)
                # join to cast_pandas.
                
                # To handle substitutions/switch-ups etc, need to separate celebs and pros.
                # then split couple into celeb/pro names
                # then join to celebs on celeb_name_part
                # then join to pros on pro_name_part.
                # But more pros than in the pros column. Need to split that field to get them all.
                
                with_names = (
                    score_pandas
                    .merge(all_pros, how='left', on='pro_name_part')
                    .merge(all_celebs, how='left', on='celeb_name_part')
                )
                
     
                problems = with_names.query('celebrity.isnull() | professional.isnull()')
                if(problems.shape[0] > 0):
                    print(problems)
                    print(all_pros)
                    print(all_celebs)

                for row in with_names.to_dict('records'):
                      yield row 
            
            
        
#         for week_tag in weekly_scores.xpath('../following-sibling::h3/span[@class="mw-headline"]'):
            
#             week = week_tag.xpath('./text()').get()
            
#             # Guest judges are just under the header, if present.
#             if (weekly_judge_candidates := week_tag.xpath("../following-sibling::p[1]/*[contains(text(), 'left to right')]")):
#                 judge_html = weekly_judge_candidates[0]
#             else:
#                 judge_html = default_judge_sentence
            
#             judge_sentence = ''.join(judge_html.xpath('./descendant-or-self::*/text()').getall())
#             score_pandas['judge_phrase'] = judge_sentence.split(':')[-1]
            
         
#             # Get score table, parse using Pandas.read_html which handles rowspan/colspan
#             score_table = week_tag.xpath('../following-sibling::table')[0]

#             score_pandas = pd.read_html(score_table.get().replace('<br>','---'))[0].fillna('') # read_html returns list of tables
#                         # remove footnote refs from column headers.
#             score_pandas = score_pandas.rename(columns=lambda x: re.sub(r'\[.*?\]','',x))
#             score_pandas = score_pandas.clean_names()
            
#             score_pandas['season'] = season
#             score_pandas['week'] = week
            
#             for row in score_pandas.to_dict('records'):
#                 yield row 
            

            
"""
Unresolved issues:
https://en.wikipedia.org/wiki/Dancing_with_the_Stars_(American_season_9)#Weekly_scores has "Dance Off" header row.  How is that handled by pandas.read_html?

Some tables have score instead of scores
e.g., https://en.wikipedia.org/wiki/Dancing_with_the_Stars_(American_season_17)#Weekly_scores week 3
https://en.wikipedia.org/wiki/Dancing_with_the_Stars_(American_season_20)#Weekly_scores Week 5.

Some tables have technical and performance scores.

https://en.wikipedia.org/wiki/Dancing_with_the_Stars_(American_season_11)#Weekly_scores  Week 4.
https://en.wikipedia.org/wiki/Dancing_with_the_Stars_(American_season_10)#Weekly_scores Week 4.

Some scores are X.  Whe
"""
