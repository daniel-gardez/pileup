# pileup
Simple code for analysing piled up pulses. The function *create_dataframe* takes a pandas DataFrame with a waveform as an 
input and returns another DataFrame with fields indicating the baseline, identification of the nearest pulses, waveform with
offset substracted and identification of the peak block if there is pileup. In order to apply it to a real scenario, these
next steps might be needed.

- Implementation of a bandstop filter before creating the dataframe.
- Fine-tuning the input values for the scipy.find_peaks function - height, prominence and length.
- Fine-tuning the random coincidences rate for finding the baseline.

We have to keep in mind that for calculating the energy of each photon count, we must fit the peak waveform. Depending on the
size of the input waveform, perhaps we might need to cut the waveform into smaller chunks, although up to several GB the
code should work, depending on the available memory.
