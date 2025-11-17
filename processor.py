import subprocess
import re
import threading
import os
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

# DÉPENDANCE NON-OPTIONNELLE:
# Whisper est maintenant requis pour l'analyse intelligente.
try:
    import whisper
except ImportError:
    whisper = None

class VideoProcessor:
    """
    Cette classe gère toute la logique de traitement vidéo, indépendamment de l'interface.
    Elle utilise des callbacks pour rapporter la progression et les logs.
    """
    def __init__(self, input_file, output_file, log_callback, progress_callback, profile="Moyen", generate_tiktoks=False):
        self.input_file = input_file
        self.output_file = output_file # Fichier de sortie (ex: highlights.mp4)
        
        # Fonctions "callback"
        self.log = log_callback
        self.update_progress = progress_callback
        
        # Options
        self.profile = profile
        self.generate_tiktoks = generate_tiktoks
        
        self.video_duration = 0.0
        self.chunk_size = 10 # Analyse par "bouts" de 10s

    def process(self):
        """
        Orchestre l'ensemble du processus de découpage vidéo.
        C'est la fonction principale à appeler dans un thread.
        """
        try:
            self.log("Démarrage du processus...")
            self.update_progress(0)
            
            if whisper is None:
                raise ImportError("Le module 'openai-whisper' est requis pour l'analyse intelligente."
                                  "Veuillez l'installer avec : pip install openai-whisper")

            # Étape 1: Obtenir la durée
            self.log("Analyse de la vidéo...")
            self.video_duration = self.get_video_duration()
            if self.video_duration == 0:
                raise Exception("Impossible d'obtenir la durée de la vidéo.")
            self.log(f"Durée totale du stream : {self.video_duration:.2f} s")
            self.update_progress(5)

            # ÉTAPE 2: Double Analyse (Cœur de la logique)
            # 2a. Analyse Sémantique (Whisper) - (5% -> 45%)
            self.log("Étape 2a: Analyse sémantique (via Whisper)...")
            self.log("(Ceci est long et identifie toutes les phrases)")
            whisper_result = self.run_transcription()
            self.log("Analyse sémantique terminée.")
            self.update_progress(45)

            # 2b. Analyse de Volume (Scoring) - (45% -> 70%)
            self.log("Étape 2b: Analyse de volume (Recherche de 'hype')...")
            chunks = self.generate_chunks(self.chunk_size)
            scored_chunks = self.score_segments_parallel(
                chunks,
                step_progress_cb=lambda p: self.update_progress(45 + p * 0.25) # 25% de la barre
            )
            self.log(f"{len(scored_chunks)} 'bouts' intenses (volume) trouvés.")
            self.update_progress(70)

            # ÉTAPE 3: Le "Cerveau" - Fusion des analyses
            self.log("Étape 3: Fusion (Cerveau) - Trouve les 'moments' intelligents...")
            intelligent_segments = self.find_intelligent_segments(whisper_result, scored_chunks)
            if not intelligent_segments:
                raise Exception("Aucun 'moment' intelligent (parole + hype) n'a été trouvé.")
            self.log(f"{len(intelligent_segments)} 'moments' intelligents identifiés.")
            self.update_progress(75)

            # ÉTAPE 4: Dérushage + Création du Highlight (75% -> 90%)
            target_duration = self.calculate_target_duration()
            self.log(f"Profil: '{self.profile}'. Durée cible: {target_duration/60:.1f} min")
            
            selected_segments = self.select_best_segments(intelligent_segments, target_duration)
            self.log(f"Sélection de {len(selected_segments)} moments pour le Highlight.")

            self.extract_and_concatenate_segments(
                selected_segments,
                step_progress_cb=lambda p: self.update_progress(75 + p * 0.15) # 15% de la barre
            )
            
            # ÉTAPE 5: Création des Tiktoks (Optionnel) (90% -> 100%)
            tiktok_moments_found = 0
            if self.generate_tiktoks:
                self.log("Étape 5: Compilation des Tiktoks (9:16)...")
                
                # NOUVELLE LOGIQUE: On compile des 'best-of' Tiktoks
                tiktok_lists = self.compile_tiktoks(intelligent_segments)
                tiktok_moments_found = len(tiktok_lists)
                
                if tiktok_moments_found > 0:
                    self.create_tiktok_clips(
                        tiktok_lists, # On passe la liste de listes
                        step_progress_cb=lambda p: self.update_progress(90 + p * 0.10) # 10% de la barre
                    )
                else:
                    self.log("Aucun 'moment' (45-75s) trouvé pour les Tiktoks.")
            
            # ÉTAPE 6: Résumé final
            self.log("\n" + ("-" * 30))
            self.log("PROCESSUS TERMINÉ AVEC SUCCÈS")
            self.log(f"  > Vidéo Highlight : {self.output_file}")
            self.log(f"  > Clips de Dérushage : '{self.get_clips_dir()}' ({len(selected_segments)} clips)")
            if self.generate_tiktoks:
                self.log(f"  > Clips Tiktok : '{self.get_tiktoks_dir()}' ({tiktok_moments_found} clips)")
            self.log(("-") * 30)
            
            self.update_progress(100)

        except Exception as e:
            self.log(f"ERREUR FATALE: {e}")
            raise e

    # --- 1. Fonctions d'Analyse (Volume + Parole) ---

    def get_video_duration(self):
        """Retourne la durée totale de la vidéo en secondes."""
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            self.input_file
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            self.log(f"Erreur ffprobe (get_video_duration): {e.stderr}")
            return 0
        except Exception:
            return 0

    def run_transcription(self):
        """Exécute Whisper sur TOUTE la vidéo."""
        try:
            self.log("Chargement du modèle de transcription 'base'...")
            self.log("(La première fois, cela téléchargera le modèle, soyez patient)")
            model = whisper.load_model("base")
            self.log("Modèle chargé. Démarrage de l'analyse sémantique...")
            result = model.transcribe(self.input_file, verbose=False, word_timestamps=True)
            return result
        except Exception as e:
            self.log(f"Erreur pendant l'analyse (Whisper): {e}")
            raise Exception(f"L'analyse sémantique a échoué. {e}")

    def generate_chunks(self, chunk_size):
        """Découpe la vidéo en 'bouts' (chunks) de 'chunk_size' secondes."""
        chunks = []
        for start in range(0, int(self.video_duration), chunk_size):
            end = start + chunk_size
            if end > self.video_duration:
                end = self.video_duration
            if end - start > 1:
                chunks.append((start, end))
        return chunks

    def _score_segment(self, start, duration):
        """Calcule le score de 'volume/hype' pour un seul segment."""
        cmd = [
            'ffmpeg', '-hide_banner', '-ss', str(start), '-t', str(duration),
            '-i', self.input_file, '-af', 'volumedetect', '-f', 'null', '-'
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            stderr = result.stderr
            m = re.search(r"max_volume: (-?\d+\.?\d*) dB", stderr)
            n = re.search(r"mean_volume: (-?\d+\.?\d*) dB", stderr)
            if m and n:
                max_vol = float(m.group(1))
                mean_vol = float(n.group(1))
                if max_vol <= -40: return 0
                vol_factor = (max_vol + 40) / 40
                dyn_factor = abs(mean_vol - max_vol) / abs(mean_vol) if mean_vol != 0 else 1
                return duration * vol_factor * dyn_factor
        except subprocess.CalledProcessError:
            return 0
        return 0

    def score_segments_parallel(self, segments, step_progress_cb):
        """Scanne les 'bouts' en parallèle pour trouver le volume."""
        scored_segments = []
        total_segments = len(segments)
        completed_count = 0
        max_workers = max(1, os.cpu_count() or 1)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._score_segment, s_start, s_end - s_start): (s_start, s_end, s_end - s_start)
                for s_start, s_end in segments
            }
            for future in as_completed(futures):
                s_start, s_end, seg_dur = futures[future]
                try:
                    score = future.result()
                    if score > 0:
                        scored_segments.append((s_start, s_end, seg_dur, score))
                except Exception as e:
                    self.log(f"Erreur de scoring sur segment {s_start}-{s_end}: {e}")
                
                completed_count += 1
                progress = (completed_count / total_segments) * 100
                step_progress_cb(progress)

        scored_segments.sort(key=lambda x: x[3], reverse=True)
        return scored_segments

    # --- 2. Fonctions "Cerveau" (Sélection & Fusion) ---

    def calculate_target_duration(self):
        """Calcule la durée cible en fonction de la durée de la vidéo et du profil."""
        INPUT_DURATION_MIN = 2 * 3600 # 2h
        INPUT_DURATION_MAX = 5 * 3600 # 5h
        
        if self.profile == "Court":
            min_target, max_target = 6 * 60, 10 * 60
        elif self.profile == "Longue":
            min_target, max_target = 20 * 60, 40 * 60
        else: # "Moyen"
            min_target, max_target = 10 * 60, 20 * 60
        
        duration = self.video_duration
        if duration <= INPUT_DURATION_MIN: return min_target
        if duration >= INPUT_DURATION_MAX: return max_target
        ratio = (duration - INPUT_DURATION_MIN) / (INPUT_DURATION_MAX - INPUT_DURATION_MIN)
        return min_target + ratio * (max_target - min_target)

    def find_intelligent_segments(self, whisper_result, scored_chunks):
        """
        LE "CERVEAU". Fusionne l'analyse de volume et l'analyse de parole.
        Ne coupe pas les phrases.
        """
        whisper_segments = whisper_result['segments']
        if not whisper_segments:
            self.log("AVERTISSEMENT: Whisper n'a détecté aucune parole. Le 'cerveau' est désactivé.")
            self.log("L'outil va se baser *uniquement* sur le volume (ancienne méthode).")
            # Mode dégradé: on retourne les chunks de volume tels quels
            return [(s[0], s[1], s[2], s[3]) for s in scored_chunks]

        # Créer une liste de (start, end) pour tous les segments de parole
        speech_timeline = [(seg['start'], seg['end']) for seg in whisper_segments]
        
        intelligent_segments = []
        # Utiliser un set pour éviter d'ajouter le même segment de parole plusieurs fois
        used_speech_segments_indices = set() 

        # Parcourir les 'bouts' les plus bruyants (déjà triés par score)
        for chunk_start, chunk_end, _, chunk_score in scored_chunks:
            
            overlapping_indices = []
            for i, (speech_start, speech_end) in enumerate(speech_timeline):
                # Vérifier s'il y a chevauchement
                if chunk_start < speech_end and chunk_end > speech_start:
                    if i not in used_speech_segments_indices:
                        overlapping_indices.append(i)

            if overlapping_indices:
                # On a trouvé de la parole dans ce "bout" bruyant !
                # Maintenant, on étend le clip pour inclure les phrases ENTIÈRES.
                
                # Trouver le début de la première phrase et la fin de la dernière
                first_index = min(overlapping_indices)
                last_index = max(overlapping_indices)
                
                new_start = speech_timeline[first_index][0]
                new_end = speech_timeline[last_index][1]
                new_duration = new_end - new_start
                
                # Ajouter tous ces segments à la liste des 'utilisés'
                for i in range(first_index, last_index + 1):
                    used_speech_segments_indices.add(i)
                
                # Ajouter ce nouveau 'moment' intelligent
                intelligent_segments.append((new_start, new_end, new_duration, chunk_score))

        # Trier les moments intelligents par score
        intelligent_segments.sort(key=lambda x: x[3], reverse=True)
        return intelligent_segments

    def select_best_segments(self, intelligent_segments, target_duration):
        """Sélectionne les meilleurs segments (intelligents) pour le Highlight."""
        selected = []
        current_total_duration = 0
        
        # Parcourir les segments (déjà triés par score)
        for start, end, duration, score in intelligent_segments:
            if current_total_duration >= target_duration:
                break
                
            # S'il reste de la place, on ajoute le segment
            # On ne coupe PAS le segment, on le prend en entier (c'est l'IA)
            selected.append((start, end, duration))
            current_total_duration += duration
            
        # Trier par ordre d'apparition pour la vidéo finale
        selected.sort(key=lambda x: x[0]) 
        return selected

    # --- 3. Fonctions de Sortie (Highlight + Dérushage) ---

    def _extract_single_segment(self, segment_info, output_filepath, pad=0.3):
        """Extrait un segment vidéo unique (encodage)."""
        # CORRECTION: segment_info peut être (start, end, duration)
        # OU (start, end, duration, score).
        # On ne prend que les deux premiers éléments par leur index.
        start = segment_info[0]
        end = segment_info[1]
        
        s_start = max(0, start - pad)
        s_duration = (end + pad) - s_start
        
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(s_start),
            '-t', str(s_duration),
            '-i', self.input_file,
            '-r', '60', '-vsync', 'cfr',
            '-c:v', 'libx264', '-preset', 'superfast', '-crf', '18',
            '-movflags', '+faststart',
            '-c:a', 'aac', '-b:a', '192k',
            '-loglevel', 'error',
            output_filepath
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"Erreur d'extraction (clip) {s_start}: {e.stderr}")
            return False

    def extract_and_concatenate_segments(self, segments, step_progress_cb, pad=0.3):
        """
        Crée le DÉRUSHAGE (clips individuels) ET le HIGHLIGHT (vidéo finale).
        """
        clips_dir = self.get_clips_dir()
        os.makedirs(clips_dir, exist_ok=True)
        self.log(f"Création du dossier de dérushage : {clips_dir}")
        self.log(f"INFO: Padding de {pad}s appliqué pour des transitions douces.")

        processed_files = []
        total_segments = len(segments)

        # 1. Dérushage (Extraction)
        for i, seg in enumerate(segments):
            start_time, _, _ = seg
            time_str = f"{int(start_time // 60)}m{int(start_time % 60):02d}s"
            clip_filename = f"clip_{i+1:03d}_{time_str}.mp4"
            clip_filepath = os.path.join(clips_dir, clip_filename)
            
            self.log(f"Dérushage - Clip {i+1}/{total_segments} ({time_str})")

            success = self._extract_single_segment(seg, clip_filepath, pad)
            if success:
                processed_files.append(clip_filepath)
            
            step_progress_cb((i + 1) / total_segments * 90) # Garde 10% pour la suite

        if not processed_files:
            raise Exception("Aucun clip n'a pu être extrait avec succès.")

        # 2. Highlight (Concaténation)
        self.log("Concaténation des clips pour le Highlight final...")
        list_file = os.path.join(clips_dir, "segments.txt")
        with open(list_file, "w", encoding='utf-8') as f:
            for seg_file in processed_files:
                f.write(f"file '{os.path.abspath(seg_file)}'\n")
        
        cmd_concat = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', list_file,
            '-c', 'copy', # Rapide, pas de ré-encodage
            self.output_file
        ]
        
        try:
            subprocess.run(cmd_concat, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
            self.log("Vidéo Highlight créée avec succès.")
        except subprocess.CalledProcessError as e:
            self.log(f"Erreur lors de la concaténation finale: {e.stderr}")
            raise Exception("La concaténation finale a échoué.")
        
        step_progress_cb(100)

    # --- 4. Fonctions de Sortie (Tiktok) ---

    def compile_tiktoks(self, intelligent_segments_pool, max_duration_per_tiktok=60, num_tiktoks_to_create=5):
        """
        NOUVELLE FONCTION "CERVEAU" TIKTOK:
        Compile des Tiktoks de ~1 minute en assemblant les MEILLEURS 
        segments intelligents (parole + hype) disponibles.
        """
        tiktoks = [] # Une liste de listes: [[segA, segB], [segC, segD, segE], ...]
        
        # Copier la liste (qui est triée par score) pour pouvoir la consommer
        segment_pool = list(intelligent_segments_pool) 

        for _ in range(num_tiktoks_to_create):
            if not segment_pool:
                break # Il n'y a plus de segments à utiliser
            
            current_tiktok_clips = []
            current_tiktok_duration = 0

            # On "remplit" le Tiktok de 60s avec les meilleurs segments disponibles
            
            # On utilise un index pour parcourir la 'pool' sans la modifier
            # pendant l'itération, ce qui est plus sûr.
            indices_to_remove = []
            
            for i, segment in enumerate(segment_pool):
                segment_duration = segment[2] # Durée
                
                # Si le clip est vide, on force l'ajout du premier (meilleur) segment
                if not current_tiktok_clips:
                    current_tiktok_clips.append(segment)
                    current_tiktok_duration += segment_duration
                    indices_to_remove.append(i)
                # Sinon, on vérifie si ça rentre
                elif current_tiktok_duration + segment_duration <= max_duration_per_tiktok:
                    current_tiktok_clips.append(segment)
                    current_tiktok_duration += segment_duration
                    indices_to_remove.append(i)
                
                # Si on est plein, on s'arrête pour ce Tiktok
                if current_tiktok_duration >= max_duration_per_tiktok:
                    break
            
            # Retirer les segments utilisés de la 'pool' (en ordre inverse pour les indices)
            for i in sorted(indices_to_remove, reverse=True):
                segment_pool.pop(i)

            if current_tiktok_clips:
                # IMPORTANT: Trier les clips par ordre chronologique
                current_tiktok_clips.sort(key=lambda x: x[0])
                tiktoks.append(current_tiktok_clips)
        
        return tiktoks


    def create_tiktok_clips(self, list_of_tiktok_lists, step_progress_cb, pad=0.3):
        """
        NOUVELLE LOGIQUE (COMPLEXE):
        Crée des Tiktoks qui sont des *compilations* de plusieurs 'cuts'.
        
        Processus pour 1 Tiktok:
        1. Extrait chaque 'cut' (ex: cut_A, cut_B) en fichier 16:9 temporaire.
        2. Concatène ces 'cuts' (temp_A + temp_B) en un fichier 16:9 (temp_concat).
        3. Applique le "crop" 9:16 sur 'temp_concat' pour créer le Tiktok final.
        4. Nettoie les fichiers temporaires.
        """
        tiktok_dir = self.get_tiktoks_dir()
        os.makedirs(tiktok_dir, exist_ok=True)
        self.log(f"Création du dossier de Tiktoks : {tiktok_dir}")

        total_tiktoks = len(list_of_tiktok_lists)
        if total_tiktoks == 0:
            step_progress_cb(100)
            return

        for i, tiktok_clips_list in enumerate(list_of_tiktok_lists):
            time_str = f"{int(tiktok_clips_list[0][0] // 60)}m{int(tiktok_clips_list[0][0] % 60):02d}s"
            self.log(f"Génération Tiktok {i+1}/{total_tiktoks} (basé sur {time_str})...")

            temp_files_for_this_tiktok = []
            temp_files_to_delete = []

            try:
                # --- 1. Extraire chaque 'cut' en fichier 16:9 temporaire ---
                for j, segment_info in enumerate(tiktok_clips_list):
                    temp_clip_path = os.path.join(tiktok_dir, f"temp_clip_{i+1}_{j+1}.mp4")
                    temp_files_to_delete.append(temp_clip_path)
                    
                    # On réutilise la fonction d'extraction 16:9
                    success = self._extract_single_segment(segment_info, temp_clip_path, pad)
                    if success:
                        temp_files_for_this_tiktok.append(temp_clip_path)

                if not temp_files_for_this_tiktok:
                    self.log(f"Avertissement: Echec de l'extraction des cuts pour Tiktok {i+1}.")
                    continue # Passer au Tiktok suivant

                # --- 2. Concaténer ces 'cuts' (toujours en 16:9) ---
                temp_concat_path = os.path.join(tiktok_dir, f"temp_concat_{i+1}.mp4")
                temp_files_to_delete.append(temp_concat_path)
                
                list_file_path = os.path.join(tiktok_dir, f"temp_list_{i+1}.txt")
                temp_files_to_delete.append(list_file_path)
                
                with open(list_file_path, "w", encoding='utf-8') as f:
                    for temp_file in temp_files_for_this_tiktok:
                        f.write(f"file '{os.path.abspath(temp_file)}'\n")
                
                cmd_concat = [
                    'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                    '-i', list_file_path,
                    '-c', 'copy', # Rapide
                    temp_concat_path
                ]
                subprocess.run(cmd_concat, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')

                # --- 3. Appliquer le "crop" 9:16 final ---
                final_tiktok_path = os.path.join(tiktok_dir, f"tiktok_{i+1:03d}_{time_str}.mp4")
                crop_filter = "crop=ih*9/16:ih,scale=1080:1920,setsar=1"
                
                cmd_crop = [
                    'ffmpeg', '-y',
                    '-i', temp_concat_path, # Entrée: le clip 16:9 concaténé
                    '-vf', crop_filter,
                    '-c:v', 'libx264', '-preset', 'superfast', '-crf', '20',
                    '-c:a', 'aac', '-b:a', '192k',
                    '-movflags', '+faststart', '-loglevel', 'error',
                    final_tiktok_path # Sortie: le clip 9:16 final
                ]
                subprocess.run(cmd_crop, check=True, capture_output=True, text=True)
                self.log(f"  > Tiktok {i+1} créé: {final_tiktok_path}")

            except subprocess.CalledProcessError as e:
                self.log(f"Erreur lors de la création du Tiktok {i+1}: {e.stderr}")
            finally:
                # --- 4. Nettoyer tous les fichiers temporaires ---
                for temp_file in temp_files_to_delete:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            
            step_progress_cb((i + 1) / total_tiktoks * 100)

    # --- 5. Fonctions "Helper" (Chemins) ---

    def get_output_name_no_ext(self):
        """Helper: 'C:/vid/highlight.mp4' -> 'highlight'"""
        return os.path.splitext(os.path.basename(self.output_file))[0]

    def get_base_output_dir(self):
        """Helper: 'C:/vid/highlight.mp4' -> 'C:/vid'"""
        return os.path.dirname(self.output_file)

    def get_clips_dir(self):
        """Dossier Dérushage: 'C:/vid/highlight_clips'"""
        return os.path.join(self.get_base_output_dir(), f"{self.get_output_name_no_ext()}_clips")

    def get_tiktoks_dir(self):
        """Dossier Tiktok: 'C:/vid/highlight_tiktoks'"""
        return os.path.join(self.get_base_output_dir(), f"{self.get_output_name_no_ext()}_tiktoks")
