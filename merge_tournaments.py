import pandas as pd
import argparse as ap

# TODO: make and print data frame showing results of hypothesis testing


class TournamentModel:

    def __init__(self, tournament, years):
        self.BASE_DATA_PATH = './data/golf/'

        self.tournament = tournament
        self.years = years

        self.tournaments, self.combined_df, self.combined_with_years_df = self._build_dfs()
        self.combined_with_years_save_slug, self.combined_save_slug = self._get_slugs()

    def _build_dfs(self, d_type='scores'):
        dfs = []
        dfs_with_years = []

        for year in self.years:
            file_path = self.BASE_DATA_PATH + self.tournament + "_" + year + "_made_cut" + ".csv"
            df = pd.read_csv(file_path)
            df.dropna(axis=0, inplace=True)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df = df[df['Type'] == d_type]
            df = df.drop(['In', 'Out'], axis=1)
            dfs.append(df)

            df_with_year = df
            df_with_year['Year'] = [year] * df.shape[0]
            dfs_with_years.append(df_with_year)

        df_with_year = pd.concat(dfs_with_years).groupby(['Player', 'Round', 'Year']).mean()
        df_with_year.reset_index(inplace=True)

        df = pd.concat(dfs_with_years).groupby(['Player', 'Round']).mean()
        df.reset_index(inplace=True)

        return dfs, df, df_with_year

    def _get_slugs(self):
        oldest_year = self.years[len(self.years) - 1]
        most_recent_year = self.years[0]
        years_slug = self.BASE_DATA_PATH + 'combined_with_years_' + self.tournament + '_' + oldest_year + '_' + most_recent_year + '.csv'
        combined_slug = self.BASE_DATA_PATH + 'combined_with_means_' + self.tournament + '_' + oldest_year + '_' + most_recent_year + '.csv'

        return years_slug, combined_slug

    def save_to_csv(self):
        self.combined_df.to_csv(self.combined_save_slug, index=False)
        self.combined_with_years_df.to_csv(self.combined_with_years_save_slug, index=False)
        pass


def parse_command_line():
    """

    :return: tournament_name, years

    """
    parser = ap.ArgumentParser()
    parser.add_argument('tournament', nargs=1, help='name of tournament')
    parser.add_argument('year', nargs='*', help='year of tournament')
    args = parser.parse_args()

    tournament_name = args.tournament[0]
    years = args.year

    return tournament_name, years


def main():
    tournament, tournament_years = parse_command_line()
    t_model = TournamentModel(tournament, tournament_years)
    t_model.save_to_csv()

    DEBUG = 2


if __name__ == '__main__':
    main()
