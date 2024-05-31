import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

def calculate_team_rankings(team_stats):
    team_stats = team_stats.sort_values(by=['Συνολικοί Βαθμοί', 'Συνολικά Γκολ Υπέρ'], ascending=[False, False])
    team_stats['Σειρά Κατάταξης'] = range(1, len(team_stats) + 1)
    return team_stats

def predict_match(home_team, away_team, team_stats, neutral_venue=False):
    home_team_data = team_stats[team_stats['Ομάδα'] == home_team]
    away_team_data = team_stats[team_stats['Ομάδα'] == away_team]

    if home_team_data.empty:
        print(f"Σφάλμα: Δεν βρέθηκαν δεδομένα για την ομάδα έδρας: {home_team}")
        return None
    if away_team_data.empty:
        print(f"Σφάλμα: Δεν βρέθηκαν δεδομένα για την ομάδα φιλοξενούμενη: {away_team}")
        return None

    home_features = home_team_data.drop(['Ομάδα', 'Σειρά Κατάταξης', 'Συνολικοί Βαθμοί'], axis=1).reset_index(drop=True)
    away_features = away_team_data.drop(['Ομάδα', 'Σειρά Κατάταξης', 'Συνολικοί Βαθμοί'], axis=1).reset_index(drop=True)

    # Check for zero values in 'Συνολικοί Αγώνες'
    if home_team_data['Συνολικοί Αγώνες'].values[0] == 0 or away_team_data['Συνολικοί Αγώνες'].values[0] == 0:
        print("Σφάλμα: Ο συνολικός αριθμός αγώνων δεν μπορεί να είναι μηδέν.")
        return None

    home_avg_goals_for = home_team_data['Συνολικά Γκολ Υπέρ'].values[0] / home_team_data['Συνολικοί Αγώνες'].values[0]
    home_avg_goals_against = home_team_data['Συνολικά Γκολ Κατά'].values[0] / home_team_data['Συνολικοί Αγώνες'].values[0]
    away_avg_goals_for = away_team_data['Συνολικά Γκολ Υπέρ'].values[0] / away_team_data['Συνολικοί Αγώνες'].values[0]
    away_avg_goals_against = away_team_data['Συνολικά Γκολ Κατά'].values[0] / away_team_data['Συνολικοί Αγώνες'].values[0]

    match_features = pd.DataFrame({
        'μέσος_όρος_γκολ_υπέρ_έδρας': [home_avg_goals_for],
        'μέσος_όρος_γκολ_κατά_έδρας': [home_avg_goals_against],
        'μέσος_όρος_γκολ_υπέρ_φιλοξενούμενη': [away_avg_goals_for],
        'μέσος_όρος_γκολ_κατά_φιλοξενούμενη': [away_avg_goals_against]
    })

    print("Χαρακτηριστικά για την ομάδα έδρας", home_team)
    print(home_team_data)
    print("Χαρακτηριστικά για την ομάδα φιλοξενούμενη", away_team)
    print(away_team_data)

    print("Χαρακτηριστικά για τον αγώνα:")
    print(match_features)

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(match_features)

    model_home_score = RandomForestRegressor(n_estimators=100, random_state=42)
    model_away_score = RandomForestRegressor(n_estimators=100, random_state=42)

    if neutral_venue:
        model_home_score.fit(X_imputed, [away_avg_goals_for])  # Use away team's average goals for neutral venue
        model_away_score.fit(X_imputed, [away_avg_goals_for])
    else:
        model_home_score.fit(X_imputed, [home_avg_goals_for])
        model_away_score.fit(X_imputed, [away_avg_goals_for])

    home_score_prediction = model_home_score.predict([X_imputed[0]])[0]
    away_score_prediction = model_away_score.predict([X_imputed[0]])[0]

    total_goals_prediction = home_score_prediction + away_score_prediction

    if home_score_prediction > away_score_prediction:
        result = f'{home_team} κερδίζει'
        outcome_code = 1
    elif home_score_prediction < away_score_prediction:
        result = f'{away_team} κερδίζει'
        outcome_code = 2
    else:
        result = 'Ισοπαλία'
        outcome_code = 'Χ'

    if result == f'{home_team} κερδίζει':
        double_chance = f'{home_team} ή Ισοπαλία'
    elif result == f'{away_team} κερδίζει':
        double_chance = f'{away_team} ή Ισοπαλία'
    else:
        double_chance = 'Ισοπαλία'

    return {
        'ομάδα_έδρας': home_team,
        'ομάδα_φιλοξενούμενη': away_team,
        'πρόβλεψη_αποτελέσματος': result,
        'διπλή_ευκαιρία': double_chance,
        'κωδικός_αποτελέσματος': outcome_code,
        'πρόβλεψη_σκορ_έδρας': round(home_score_prediction, 2),
        'πρόβλεψη_σκορ_φιλοξενούμενης': round(away_score_prediction, 2),
        'πρόβλεψη_συνολικών_γκολ': round(total_goals_prediction, 2)
    }

def process_file(file_path, is_league):
    try:
        # Load the file with proper delimiter detection
        delimiter = '\t' if 'other_tournaments' in file_path else ','
        team_stats = pd.read_csv(file_path, delimiter=delimiter, header=0)
        
        # Display the first few rows to understand the structure
        print("Πρώτες γραμμές από το αρχείο:")
        print(team_stats.head())

        # Ensuring columns are correct and adding if necessary
        expected_columns = ['Ομάδα', 'Συνολικοί Βαθμοί', 'Σειρά Κατάταξης', 'Συνολικά Γκολ Υπέρ', 'Συνολικά Γκολ Κατά', 'Συνολικοί Αγώνες']
        for column in expected_columns:
            if column not in team_stats.columns:
                team_stats[column] = 0

        team_stats = calculate_team_rankings(team_stats)
        print("Διαθέσιμες ομάδες:")
        print(team_stats['Ομάδα'].tolist())

        home_team = input("Δώσε το όνομα της ομάδας έδρας: ")
        away_team = input("Δώσε το όνομα της ομάδας φιλοξενούμενης: ")

        if is_league:
            match_prediction = predict_match(home_team, away_team, team_stats)
            if match_prediction:
                print(match_prediction)
        else:
            neutral_venue_input = input("Είναι η έδρα ουδέτερη; (ναι/όχι): ").strip().lower()
            neutral_venue = neutral_venue_input == 'ναι'

            print("Στατιστικά για την ομάδα έδρας:")
            print(team_stats[team_stats['Ομάδα'] == home_team])
            print("Στατιστικά για την ομάδα φιλοξενούμενη:")
            print(team_stats[team_stats['Ομάδα'] == away_team])
            
            match_prediction = predict_match(home_team, away_team, team_stats, neutral_venue)
            if match_prediction:
                print(match_prediction)
    except pd.errors.EmptyDataError:
        print("Σφάλμα: Δεν υπάρχουν στήλες για ανάλυση από το αρχείο, το αρχείο μπορεί να είναι κενό ή ακατάλληλα μορφοποιημένο.")
    except Exception as e:
        print(f"Παρουσιάστηκε ένα απροσδόκητο σφάλμα: {e}")

def main():
    print("Τρέχουσα διαδρομή εργασίας:", os.getcwd())
    choice = input("Επιλέξτε 1 για Πρωτάθλημα ή 2 για Άλλες Διοργανώσεις: ")

    if choice == '1':
        file_path = "/Users/chrysovalantistsiartas/Desktop/bets/team_stats_league.csv"
    elif choice == '2':
        file_path = "/Users/chrysovalantistsiartas/Desktop/bets/team_stats_other_tournaments.csv"
    else:
        print("Μη έγκυρη επιλογή. Παρακαλώ επιλέξτε 1 ή 2.")
        return

    print(f"Using file path: {file_path}")
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return

    is_league = choice == '1'
    process_file(file_path, is_league)

if __name__ == "__main__":
    main()
