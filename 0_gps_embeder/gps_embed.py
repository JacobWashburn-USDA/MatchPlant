import csv 
import os 
import shutil
import piexif 
from PIL import Image 
from typing import Dict, List, Optional
from pyproj import Transformer 
from datetime import datetime 

class TimestampMatcher:
    """TimestampMatcher processes drone images and matches them with GPS coordinates."""
    def __init__(self):
        """Initialize the TimestampMatcher with empty data structers"""
        self.events_data = []
        self.time_offset = None
        self.matched_events = {}
        self.match_errors = {}
        self.matched_results = []
        self.utm_zone = None
        self.hemisphere = None

    #-------------------
    # 1. Setup Methods
    #-------------------

    def set_utm_zone(self, zone: int, hemisphere: str) -> None:
        """Set UTM zone and hemishere for coordinate conversion"""
        if not (1 <= zone <= 60):
            raise ValueError("UTM zone must be between 1 and 60")
        if hemisphere.upper() not in ['N', 'S']:
            raise ValueError("Hemisphere must be 'N' or 'S'")
        
        self.utm_zone = zone
        self.hemisphere = hemisphere.upper()
        print(f"Set UTM zone: {self.utm_zone}{self.hemisphere}")

    def get_utm_crs(self) -> str:
        """Get UTM CRS string for coordinate transformation"""
        if self.utm_zone is None or self.hemisphere is None:
            raise ValueError("UTM zone and hemisphere must be set first")
        
        base = 32600 if self.hemisphere == 'N' else 32700
        epsg = base + self.utm_zone 
        return f'EPSG:{epsg}'

    #-----------------------
    # 2. Input Processing
    #-----------------------

    def get_image_files(self, folder_path: str) -> List[str]:
        """Get all image files from folder sorted by name"""
        image_extensions = {'.jpg', '.jpeg', '.JPG', '.JPEG'}
        image_files = []

        for filename in os.listdir(folder_path):
            if not filename.startswith("._") and os.path.splitext(filename)[1] in image_extensions:
                full_path = os.path.join(folder_path, filename)
                image_files.append(full_path)
        
        # Sort images based on numeric parts of theor flenames
        image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))

        print(f"Found {len(image_files)} images in {folder_path}")
        if  image_files:
            print(f"First image: {os.path.basename(image_files[0])}")
            print(f"Last image: {os.path.basename(image_files[-1])}")

        return image_files
    
    def load_events_file(self, file_path: str) -> None:
        """Load and parse events file"""
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            self.events_data = [] 

            for row in reader:
                if row['Timestamp'].strip():
                    timestamp = float(row["Timestamp"])
                    self.events_data.append({
                        'timestamp': timestamp, 
                        'data': row
                        })
        print(f"Loaded {len(self.events_data)} events")
        print(f"Using UTM Zone: {self.utm_zone}{self.hemisphere}")

    #-----------------------------
    # 3. Time Matching Process
    #-----------------------------

    def read_image_exif(self, image_path: str) -> Optional[Dict]:
        """Read EXIF data from an image"""
        with Image.open(image_path) as img:
            exif = img._getexif()
            if exif is None:
                print(f"No EXIF data in {os.path.basename(image_path)}")
                return None 
        
        datetime_tag = exif.get(306) # 306 is DateTime tag 
        if not datetime_tag:
            print(f"No DateTime in EXIF for {os.path.basename(image_path)}")
            return None 
        
        return {
            'DateTime': datetime_tag,
            'SubSecTime': exif.get(37520)
        }
    
    def convert_exif_timestamp(self, timestamp_str: str) -> float:
        """Convert EXIF timestamp to float"""
        dt = datetime.strptime(timestamp_str, "%Y:%m:%d %H:%M:%S")
        return dt.timestamp()
    
    def find_initial_offset(self, reference_image: str) -> Optional[float]:
        """Calculate initial time offset from reference image"""
        if not self.events_data:
            raise ValueError("No events data loaded")
        
        exif_data = self.read_image_exif(reference_image)
        if not exif_data:
            return None
        
        exif_seconds = self.convert_exif_timestamp(exif_data['DateTime'])
        event_times = [event['timestamp'] for event in self.events_data]
        min_time = min(event_times)

        self.time_offset = exif_seconds - min_time
        print(f"Using {os.path.basename(reference_image)} as a reference image")
        return self.time_offset
    
    def find_matching_event(self, image_path: str, tolerence: float = 2.0) -> Optional[Dict]:
        """Find matching event for an image within time tolerance"""
        if self.time_offset is None:
            raise ValueError("Time offset not calculated")
        
        exif_data = self.read_image_exif(image_path)
        if not exif_data:
            return None 
        
        exif_seconds = self.convert_exif_timestamp(exif_data['DateTime'])
        target_time = exif_seconds - self.time_offset

        closest_event = min(self.events_data, key=lambda x: abs(x['timestamp'] - target_time))
        time_diff = abs(closest_event['timestamp'] - target_time)

        if time_diff <= tolerence:
            event_data = closest_event['data']
            self.matched_events[closest_event['timestamp']] = os.path.basename(image_path)

            # Store error information 
            percent_error = (time_diff / closest_event['timestamp']) * 100 
            self.match_errors[closest_event['timestamp']] = {
                'image': os.path.basename(image_path),
                'exif_time': exif_seconds,
                'target_time': target_time,
                'event_time': closest_event['timestamp'],
                'diff_seconds': time_diff,
                'percent_error': percent_error
            }

            # Convert coordinates if available 
            if all(key in event_data for key in ['X', 'Y', 'Z']):
                coords = self.convert_coordinates(
                    float(event_data['X']),
                    float(event_data['Y']),
                    float(event_data['Z'])
                )
                self.matched_results.append({
                    'image': os.path.basename(image_path),
                    **coords
                })

            return event_data
        return None
    
    #-----------------------------
    # 4. Coordinate Processing
    #-----------------------------

    def convert_coordinates(self, x: float, y: float, z: float) -> Dict:
        """Convert from UTM to WGS84 coordinates"""
        if self.utm_zone is None:
            raise ValueError("UTM zone must be set before converting coordinates")
        
        utm_crs = self.get_utm_crs()
        transformer = Transformer.from_crs(utm_crs, 'EPSG:4326', always_xy=True)

        lon, lat = transformer.transform(x, y)

        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError(f"Invalid converstion result: lon={lon}, lat{lat}")
        
        return {
            'latitude': round(lat, 7), # About 1cm precision
            'longitude': round(lon, 7), 
            'altitude': round(z, 2)
        }

    #--------------------------
    # 5. Output Generation
    #--------------------------

    def save_matched_events(self, input_file: str, output_file: str):
        """Save events data with matched images and timing errors"""
        with open(input_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            headers = next(reader)

        new_headers = headers + ['Matched_Image', 'Time_Difference_Seconds', 'Percent_Error']
        matched_count = 0
        total_error = 0 
        max_error = 0 
        max_error_image = ''

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=new_headers, delimiter=',')
            writer.writeheader()

            for event in self.events_data:
                row = {k: v for k, v in event['data'].items() if k is not None}
                timestamp = event['timestamp']

                if timestamp in self.match_errors:
                    matched_count += 1
                    error_info = self.match_errors[timestamp]

                    row['Matched_Image'] = error_info['image']
                    row['Time_Difference_Seconds'] = f"{error_info['diff_seconds']:.6f}"
                    row['Percent_Error'] = f"{error_info['percent_error']:.6f}"

                    total_error += error_info['percent_error']
                    if error_info['percent_error'] > max_error:
                        max_error = error_info['percent_error']
                        max_error_image = error_info['image']
                    
                else:
                    row['Matched_Image'] = ''
                    row['Time_Difference_Seconds'] = ''
                    row['Percent_Error'] = ''

                writer.writerow(row)

        if matched_count > 0:
            avg_error = total_error / matched_count 
            print(f"\nError Analysis:")
            print(f"Average error: {avg_error:.6f}%")
            print(f"Maximum error: {max_error:.6f}% (in {max_error_image})")

        print(f"\nMatching Results:")
        print(f"Total events: {len(self.events_data)}")
        print(f"Matched images: {matched_count}")
        print(f"Output saved to: {output_file}")

    def embed_gps_to_images(self, input_folder: str, output_folder: str) -> None:
        """Embed GPS coordinates into matched images"""
        os.makedirs(output_folder, exist_ok=True)

        print(f"\nEmbedding GPS data into images...")
        print(f"Found {len(self.matched_results)} images with GPS data")

        for match in self.matched_results:
            source_image = os.path.join(input_folder, match['image'])
            if not os.path.exists(source_image):
                print(f"Warning: Source image not found: {match['image']}")
                continue

            output_path = os.path.join(output_folder, match['image'])

            # Copy original file
            shutil.copy2(source_image, output_path)

            # Write  GPS data
            gps_data = {
                'latitude': match['latitude'],
                'longitude': match['longitude'],
                'altitude': match['altitude']
            }
            self._write_gps_to_exif(output_path, gps_data)
            print(f"Processed {match['image']}")

        print(f"\nGPS embedding complete")
        print(f"Tagged images saves to: {output_folder}")
    
    def _write_gps_to_exif(self, image_path: str, gps_data: Dict):
        """Write GPS data to image EXIF metadata"""
        exif_dict = piexif.load(image_path)
        if not exif_dict:
            exif_dict = {
                '0th': {}, 
                'Exif': {}, 
                'GPS': {}, 
                '1st': {}, 
                'thumbnail': None
            }

        # Convert coordinates to DMS with high precision
        lat_deg = abs(gps_data["latitude"])
        lon_deg = abs(gps_data["longitude"])

        lat_d = int(lat_deg)
        lat_m = int((lat_deg - lat_d) * 60)
        lat_s = int(((lat_deg - lat_d) * 60 - lat_m) * 60 * 1000000)

        lon_d = int(lon_deg)
        lon_m = int((lon_deg - lon_d) * 60)
        lon_s = int(((lon_deg - lon_d) * 60 - lon_m) * 60 * 1000000)

        # Update GPS IFD
        gps_ifd = {
            piexif.GPSIFD.GPSVersionID: (2, 2, 0, 0),
            piexif.GPSIFD.GPSLatitudeRef: 'N' if gps_data['latitude'] >= 0 else 'S',
            piexif.GPSIFD.GPSLatitude: [
                (lat_d, 1), 
                (lat_m, 1), 
                (lat_s, 1000000)
            ],
            piexif.GPSIFD.GPSLongitudeRef: 'E' if gps_data['longitude'] >= 0 else 'W',
            piexif.GPSIFD.GPSLongitude: [
                (lon_d, 1), 
                (lon_m, 1), 
                (lon_s, 1000000)
            ]
        }

        # Add altitude if provided
        if gps_data['altitude'] is not None:
            alt = abs(gps_data['altitude'])
            gps_ifd[piexif.GPSIFD.GPSAltitudeRef] = 1 if gps_data['altitude'] < 0 else 0
            gps_ifd[piexif.GPSIFD.GPSAltitude] = (int(alt * 1000), 1000)

        exif_dict['GPS'] = gps_ifd

        # Remove problematic EXIF tags if present 
        if 'Exif' in exif_dict:
            problematic_tags = [37121]
            for tag in problematic_tags:
                if tag in exif_dict['Exif']:
                    del exif_dict['Exif'][tag]

        # Convert EXIF data to bytes and save 
        exif_bytes = piexif.dump(exif_dict)
        with Image.open(image_path) as img:
            img.save(image_path, 'JPEG', exif=exif_bytes, quality='keep')

        print(f"Successfully updated GPS data in {os.path.basename(image_path)}")
        print(f"GPS: {gps_data['latitude']}, {gps_data['longitude']}, {gps_data['altitude']}")

#---------------------
# Helper Functions
#---------------------

def get_user_utm_input() -> tuple:
    """Get UTM zone information from user with validation"""
    while True:
        try:
            zone = int(input("\nEnter UTM zone number (1-60): "))
            if not 1 <= zone <= 60:
                print("Error: Zone must be betweeen 1 and 60")
                continue

            hemisphere = input("Enter hemisphere (N/S): ").upper()
            if hemisphere not in ['N', 'S']:
                print("Error: Hemisphere must be N or S")
                continue

            return zone, hemisphere
        
        except ValueError:
            print("Error: Please enter a valid number for zone")
            continue

def main():
    """Main execution function implementing the processing pipeline"""
    # 1. Initialize matcher
    matcher = TimestampMatcher()

    # 2. Setup: Get UTM zone from user
    print("Please specify the UTM zone for your coordinates:")
    zone, hemisphere = get_user_utm_input()
    matcher.set_utm_zone(zone, hemisphere)

    # 3. Setup path
    base_folder = input("\nEnter the path to your images folder: ")
    if not os.path.exists(base_folder):
        raise ValueError(f"Folder not found: {base_folder}")
    
    events_file = os.path.join(base_folder, 'events.txt')
    output_file = os.path.join(base_folder, 'events_with_matches.csv')
    output_folder = os.path.join(base_folder, 'images_with_gps')

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # 4. Input Processing: Get images and load events
    print(f"\nSearching for images in: {base_folder}")
    image_files = matcher.get_image_files(base_folder)

    if not image_files:
        raise ValueError("No image found in the specified folder")
    
    print(f"\nReading events file: {events_file}")
    matcher.load_events_file(events_file)

    # 5. Time Matching: Calculate offset and process images
    if not matcher.find_initial_offset(image_files[0]):
        raise ValueError("Could not calculate time offset from reference image")
    
    print("\nProcessing images...")
    total_images = len(image_files)
    matched_count = 0

    for i, image_path in enumerate(image_files, 1):
        match = matcher.find_matching_event(image_path)
        if match:
            matched_count += 1
            if i % 10 == 0: # Progress updates every 10 images
                print(f"Progress: {i}/{total_images} images processed ({matched_count} matched found)")

    # 6. Output Generation: Save results and embed GPS
    print(f"\nSaving matching results to: {output_file}")
    matcher.save_matched_events(events_file, output_file)

    print(f"\nEmbedding GPS data in images...")
    matcher.embed_gps_to_images(base_folder, output_folder)

if __name__ == "__main__":
    main()
