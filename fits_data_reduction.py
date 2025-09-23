import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.time import Time, TimeDelta


# Collect Source Names from the catalog
def source_names(catalog_data):
    source_name = []
    for sources in catalog_data:
        # print(len(sources['Flux_History']))
        source_name.append(sources[0])
    return source_name

# Open the FITS file
file_name = './Data/gll_psc_v35.fit'
hdul = fits.open(file_name)
hdul.info()
# Get the data for the sources ---
# Source catalog headers/features
data_headers = hdul[1].columns.names
print(len(data_headers), data_headers)
catalog_data = hdul[1].data
# classification_list = []
bll = 0
fsrq = 0
bcu = 0
for i,e in enumerate(catalog_data):
    if e['CLASS1']=='fsrq':
        fsrq += 1
    if e['CLASS2']=='fsrq':
        fsrq += 1
    if e['CLASS1']=='bll':
        bll += 1
    if e['CLASS2']=='bll':
        bll += 1
    if e['CLASS1']=='bcu':
        bcu += 1
    if e['CLASS2']=='bcu':
        bcu += 1
    # classification_list.append(e['CLASS1'])
    # classification_list.append(e['CLASS2'])
print(bll, fsrq, bcu)
source_name_lst = source_names(catalog_data)
print(f"Names of {len(catalog_data)} sources:")
# print(source_name_lst)
first_source = catalog_data[2]
print(first_source['CLASS1'], first_source['CLASS2'])
source_name = first_source['Source_Name']

# --- Extract the flux, uncertainty, and TS history arrays ---
flux_history = first_source['Flux_History']
unc_flux_history = first_source['Unc_Flux_History']
ts_history = first_source['Sqrt_TS_History']**2

# --- Get the time bin information ---
hist_start_hdu = hdul[6]
time_column_name = hist_start_hdu.columns.names[0]
time_start_met = hist_start_hdu.data[time_column_name]
# print("Time bin start times:", time_start_met)
# Convert MET to MJD using the reference time from the header
mjdreff_str = hdul[1].header['MJDREFF']
mjdreff_float = float(mjdreff_str.replace('D', 'e'))

met_ref_mjd = hdul[1].header['MJDREFI'] + mjdreff_float
met_ref_time = Time(met_ref_mjd, format='mjd', scale='tt')

times = met_ref_time + TimeDelta(time_start_met, format='sec')

# --- Slice the time array to match the length of the data arrays
times_mjd = times.mjd[:len(flux_history)]

# --- Separate detections from upper limits and get flux values ---
detection_threshold = 4
is_detection = ts_history >= detection_threshold

flux_detections = flux_history[is_detection]
flux_upper_limits = flux_history[~is_detection]

# Use slicing to get the lower and upper uncertainty arrays
unc_lower = unc_flux_history[:, 0]
unc_upper = unc_flux_history[:, 1]

unc_lower_detections = unc_lower[is_detection]
unc_upper_detections = unc_upper[is_detection]
y_err_detections = np.array([np.abs(unc_lower_detections), unc_upper_detections])

upper_limit_values = unc_upper[~is_detection]

time_detections = times_mjd[is_detection]
time_upper_limits = times_mjd[~is_detection]

# --- Create the light curve plot ---
fig, ax = plt.subplots(figsize=(12, 6))

# Plot detections with error bars
ax.errorbar(
    time_detections,
    flux_detections,
    yerr=y_err_detections,
    fmt='o',
    capsize=3,
    color='blue',
    label='Detections (TS > 4)'
)

# Plot upper limits with arrows
ax.errorbar(
    time_upper_limits,
    upper_limit_values,
    yerr=upper_limit_values * 0.5,
    uplims=True,
    fmt='v',
    capsize=0,
    color='red',
    label='95% CL Upper Limits'
)

# Set plot labels and title
ax.set_xlabel('Time (MJD)')
ax.set_ylabel(r'Flux ($>1\ GeV\ ph\ cm^{-2}\ s^{-1}$)')
ax.set_title(f'Light Curve for {source_name}')
ax.set_yscale('log')
ax.grid(True, which="both", ls="--", alpha=0.5)
ax.legend()
plt.tight_layout()
# plt.savefig(f'./LCs/{source_name}.png')
plt.show()

# Close the FITS file
hdul.close()

print(f"Light curve plot saved as {source_name}_LC.png")