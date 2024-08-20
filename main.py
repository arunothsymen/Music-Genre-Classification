import pickle
import streamlit as st
import pandas as pd

# Function to load the pre-trained model
def load_model():
    try:
        with open('model3.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Main Page
st.title("Music Genre Prediction")
st.markdown("""
Upload a CSV file containing music features, and the model will predict the genre based on the input data.
""")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    
    # Define required columns (make sure to include all necessary columns here)
    required_columns = [
        "highlevel_danceability_value", "highlevel_equalization_profile_value", "highlevel_excitement_value", "highlevel_intensity_value",
        "loudness_dynamic_complexity_dvar", "loudness_dynamic_complexity_mean", "loudness_dynamic_complexity_var", "loudness_larm_dvar",
        "loudness_larm_mean", "loudness_larm_var", "loudness_replay_gain_value", "loudness_vicker_dvar", "loudness_vicker_mean",
        "loudness_vicker_var", "rhythm_beats_loudness_bass_dvar", "rhythm_beats_loudness_bass_mean", "rhythm_beats_loudness_bass_var",
        "rhythm_beats_loudness_dvar", "rhythm_beats_loudness_mean", "rhythm_beats_loudness_var", "rhythm_onset_rate_value",
        "spectral_barkbands_00_dvar", "spectral_barkbands_00_mean", "spectral_barkbands_00_var", "spectral_barkbands_01_dvar",
        "spectral_barkbands_01_mean", "spectral_barkbands_01_var", "spectral_barkbands_02_dvar", "spectral_barkbands_02_mean",
        "spectral_barkbands_02_var", "spectral_barkbands_03_dvar", "spectral_barkbands_03_mean", "spectral_barkbands_03_var",
        "spectral_barkbands_04_dvar", "spectral_barkbands_04_mean", "spectral_barkbands_04_var", "spectral_barkbands_05_dvar",
        "spectral_barkbands_05_mean", "spectral_barkbands_05_var", "spectral_barkbands_06_dvar", "spectral_barkbands_06_mean",
        "spectral_barkbands_06_var", "spectral_barkbands_07_dvar", "spectral_barkbands_07_mean", "spectral_barkbands_07_var",
        "spectral_barkbands_08_dvar", "spectral_barkbands_08_mean", "spectral_barkbands_08_var", "spectral_barkbands_09_dvar",
        "spectral_barkbands_09_mean", "spectral_barkbands_09_var", "spectral_barkbands_10_dvar", "spectral_barkbands_10_mean",
        "spectral_barkbands_10_var", "spectral_barkbands_11_dvar", "spectral_barkbands_11_mean", "spectral_barkbands_11_var",
        "spectral_barkbands_12_dvar", "spectral_barkbands_12_mean", "spectral_barkbands_12_var", "spectral_barkbands_13_dvar",
        "spectral_barkbands_13_mean", "spectral_barkbands_13_var", "spectral_barkbands_14_dvar", "spectral_barkbands_14_mean",
        "spectral_barkbands_14_var", "spectral_barkbands_15_dvar", "spectral_barkbands_15_mean", "spectral_barkbands_15_var",
        "spectral_barkbands_16_dvar", "spectral_barkbands_16_mean", "spectral_barkbands_16_var", "spectral_barkbands_17_dvar",
        "spectral_barkbands_17_mean", "spectral_barkbands_17_var", "spectral_barkbands_18_dvar", "spectral_barkbands_18_mean",
        "spectral_barkbands_18_var", "spectral_barkbands_19_dvar", "spectral_barkbands_19_mean", "spectral_barkbands_19_var",
        "spectral_barkbands_20_dvar", "spectral_barkbands_20_mean", "spectral_barkbands_20_var", "spectral_barkbands_21_dvar",
        "spectral_barkbands_21_mean", "spectral_barkbands_21_var", "spectral_barkbands_22_dvar", "spectral_barkbands_22_mean",
        "spectral_barkbands_22_var", "spectral_barkbands_23_dvar", "spectral_barkbands_23_mean", "spectral_barkbands_23_var",
        "spectral_barkbands_24_dvar", "spectral_barkbands_24_mean", "spectral_barkbands_24_var", "spectral_barkbands_25_dvar",
        "spectral_barkbands_25_mean", "spectral_barkbands_25_var", "spectral_barkbands_kurtosis_dvar", "spectral_barkbands_kurtosis_mean",
        "spectral_barkbands_kurtosis_var", "spectral_barkbands_skewness_dvar", "spectral_barkbands_skewness_mean",
        "spectral_barkbands_skewness_var", "spectral_barkbands_spread_dvar", "spectral_barkbands_spread_mean", "spectral_barkbands_spread_var",
        "spectral_centroid_dvar", "spectral_centroid_mean", "spectral_centroid_var", "spectral_crest_dvar", "spectral_crest_mean",
        "spectral_crest_var", "spectral_decrease_dvar", "spectral_decrease_mean", "spectral_decrease_var", "spectral_energy_dvar",
        "spectral_energy_mean", "spectral_energy_var", "spectral_energybandratio_high_dvar", "spectral_energybandratio_high_mean",
        "spectral_energybandratio_high_var", "spectral_energybandratio_low_dvar", "spectral_energybandratio_low_mean",
        "spectral_energybandratio_low_var", "spectral_energybandratio_middle_high_dvar", "spectral_energybandratio_middle_high_mean",
        "spectral_energybandratio_middle_high_var", "spectral_energybandratio_middle_low_dvar", "spectral_energybandratio_middle_low_mean",
        "spectral_energybandratio_middle_low_var", "spectral_flatness_db_dvar", "spectral_flatness_db_mean", "spectral_flatness_db_var",
        "spectral_flux_dvar", "spectral_flux_mean", "spectral_flux_var", "spectral_hfc_dvar", "spectral_hfc_mean", "spectral_hfc_var",
        "spectral_kurtosis_dvar", "spectral_kurtosis_mean", "spectral_kurtosis_var", "spectral_mfcc_00_dvar", "spectral_mfcc_00_mean",
        "spectral_mfcc_00_var", "spectral_mfcc_01_dvar", "spectral_mfcc_01_mean", "spectral_mfcc_01_var", "spectral_mfcc_02_dvar",
        "spectral_mfcc_02_mean", "spectral_mfcc_02_var", "spectral_mfcc_03_dvar", "spectral_mfcc_03_mean", "spectral_mfcc_03_var",
        "spectral_mfcc_04_dvar", "spectral_mfcc_04_mean", "spectral_mfcc_04_var", "spectral_mfcc_05_dvar", "spectral_mfcc_05_mean",
        "spectral_mfcc_05_var", "spectral_mfcc_06_dvar", "spectral_mfcc_06_mean", "spectral_mfcc_06_var", "spectral_mfcc_07_dvar",
        "spectral_mfcc_07_mean", "spectral_mfcc_07_var", "spectral_mfcc_08_dvar", "spectral_mfcc_08_mean", "spectral_mfcc_08_var",
        "spectral_mfcc_09_dvar", "spectral_mfcc_09_mean", "spectral_mfcc_09_var", "spectral_mfcc_10_dvar", "spectral_mfcc_10_mean",
        "spectral_mfcc_10_var", "spectral_mfcc_11_dvar", "spectral_mfcc_11_mean", "spectral_mfcc_11_var", "spectral_mfcc_12_dvar",
        "spectral_mfcc_12_mean", "spectral_mfcc_12_var", "spectral_pitch_histogram_spread_value", "spectral_pitch_instantaneous_confidence_dvar",
        "spectral_pitch_instantaneous_confidence_mean", "spectral_pitch_instantaneous_confidence_var", "spectral_pitch_salience_dvar",
        "spectral_pitch_salience_mean", "spectral_pitch_salience_var", "spectral_rms_dvar", "spectral_rms_mean", "spectral_rms_var",
        "spectral_rolloff_dvar", "spectral_rolloff_mean", "spectral_rolloff_var", "spectral_silence_rate_20dB_dvar",
        "spectral_silence_rate_20dB_mean", "spectral_silence_rate_20dB_var", "spectral_silence_rate_30dB_dvar",
        "spectral_silence_rate_30dB_mean", "spectral_silence_rate_30dB_var", "spectral_silence_rate_60dB_dvar",
        "spectral_silence_rate_60dB_mean", "spectral_silence_rate_60dB_var", "spectral_skewness_dvar", "spectral_skewness_mean",
        "spectral_skewness_var", "spectral_spread_dvar", "spectral_spread_mean", "spectral_spread_var", "spectral_strongpeak_dvar",
        "spectral_strongpeak_mean", "spectral_strongpeak_var", "temporal_zerocrossingrate_dvar", "temporal_zerocrossingrate_mean",
        "temporal_zerocrossingrate_var", "tempotap_bpm_estimates_dvar", "tempotap_bpm_estimates_mean", "tempotap_bpm_estimates_var",
        "tempotap_bpm_value", "timbral_complexity_dvar", "timbral_complexity_mean", "timbral_complexity_var", "tonal_chords_changes_rate_value",
        "tonal_chords_dissonance_dvar", "tonal_chords_dissonance_mean", "tonal_chords_dissonance_var", "tonal_chords_number_rate_value",
        "tonal_chords_strength_dvar", "tonal_chords_strength_mean", "tonal_chords_strength_var", "tonal_dissonance_dvar",
        "tonal_dissonance_mean", "tonal_dissonance_var", "tonal_key_strength_value", "tonal_tuning_equal_tempered_deviation_value"
    ]
    
    # Check if all required columns are present
    if all(column in data.columns for column in required_columns):
        model = load_model()
        
        if model is not None:
            # Make predictions
            try:
                predictions = model.predict(data[required_columns])
                data['Predicted Genre'] = predictions

                # Keep only the 'Genre' and 'Predicted Genre' columns
                if 'Genre' in data.columns:
                    result_data = data[['Genre', 'Predicted Genre']]
                else:
                    result_data = data[['Predicted Genre']]

                # Display results and image
                col1, col2 = st.columns([3, 1])  # Adjust the ratio as needed
                with col1:
                    st.write("Predictions:")
                    st.write(result_data)
                    
                    # Option to download the prediction results
                    csv = result_data.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
                    
                with col2:
                    st.image("genres.jpg", caption="Music Genres", use_column_width=True)
                    
            except Exception as e:
                st.error(f"Error making predictions: {e}")
    else:
        st.error("The uploaded file does not contain all required columns. Please check the input data.")
else:
    st.info("Please upload a CSV file to proceed.")
