import os
import shutil

def process_labels_and_images(base_dir):
    # Parcours des dossiers train, valid, test
    for split in ['train', 'valid', 'test']:
        labels_dir = os.path.join(base_dir, split, 'labels')
        images_dir = os.path.join(base_dir, split, 'images')
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue

            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, 'r') as f:
                lines = f.readlines()

            # On ne garde que les lignes de classe 0 ou 2
            kept_lines = [line for line in lines if line.strip() and line.split()[0] in ('0', '2')]

            if kept_lines:
                # Si au moins une ligne à garder, on réécrit le fichier avec uniquement les classes 0 ou 2
                with open(label_path, 'w') as f:
                    f.writelines(kept_lines)
            else:
                # Sinon, on supprime le fichier label et l'image correspondante
                os.remove(label_path)
                # Cherche l'image correspondante (peut être .jpg, .jpeg, .png...)
                base_name = os.path.splitext(label_file)[0]
                # Cherche les fichiers image avec le même nom de base
                for ext in ['.jpg', '.jpeg', '.png']:
                    image_path = os.path.join(images_dir, base_name + ext)
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        break  # On arrête après avoir trouvé/supprimé une image


def replace_class(base_dir, replaced_class, new_class):
    for split in ['train', 'valid', 'test']:
        labels_dir = os.path.join(base_dir, split, 'labels')
        for filename in os.listdir(labels_dir):
            if not filename.endswith('.txt'):
                continue
            label_path = os.path.join(labels_dir, filename)
            with open(label_path, 'r') as f:
                lines = f.readlines()

            # Remplacement de la classe
            new_lines = []
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if parts[0] == replaced_class:
                        parts[0] = new_class
                    new_lines.append(' '.join(parts) + '\n')

            with open(label_path, 'w') as f:
                f.writelines(new_lines)

import os
import shutil

def clone_dataset(src_dataset_path, dst_dataset_path):
    """
    Clone les images et labels du dataset source (src_dataset_path) dans le dataset cible (dst_dataset_path).
    Fusionne les splits 'train', 'valid', 'test'.
    """
    for split in ['train', 'valid', 'test']:
        for subdir in ['images', 'labels']:
            src_dir = os.path.join(src_dataset_path, split, subdir)
            dst_dir = os.path.join(dst_dataset_path, split, subdir)
            os.makedirs(dst_dir, exist_ok=True)
            for filename in os.listdir(src_dir):
                src_file = os.path.join(src_dir, filename)
                dst_file = os.path.join(dst_dir, filename)
                if os.path.isfile(src_file):
                    # Pour éviter l'écrasement, on peut renommer en cas de conflit
                    if os.path.exists(dst_file):
                        base, ext = os.path.splitext(filename)
                        i = 1
                        new_filename = f"{base}_clone{i}{ext}"
                        while os.path.exists(os.path.join(dst_dir, new_filename)):
                            i += 1
                            new_filename = f"{base}_clone{i}{ext}"
                        dst_file = os.path.join(dst_dir, new_filename)
                    shutil.copy2(src_file, dst_file)


def delete_clone1_files(dataset_path):
    """
    Supprime tous les fichiers dont le nom se termine par '_clone1' (avant l'extension)
    dans les dossiers images/labels de train, valid, test.
    """
    for split in ['valid', 'train', 'test']:
        for subdir in ['images', 'labels']:
            target_dir = os.path.join(dataset_path, split, subdir)
            if not os.path.exists(target_dir):
                continue
            for filename in os.listdir(target_dir):
                base, ext = os.path.splitext(filename)
                if base.endswith('_clone1'):
                    file_path = os.path.join(target_dir, filename)
                    os.remove(file_path)
                    print(f"Supprimé : {file_path}")


import os

def sync_labels_and_images(dataset_path):
    """
    Supprime les fichiers labels sans image correspondante et, optionnellement,
    les images sans label correspondant dans les dossiers train, valid, test.
    """
    cptLabel = 0
    cptImage = 0
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(dataset_path, split, 'images')
        lbl_dir = os.path.join(dataset_path, split, 'labels')
        if not (os.path.exists(img_dir) and os.path.exists(lbl_dir)):
            continue

        image_bases = set(os.path.splitext(f)[0] for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)))
        label_bases = set(os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if os.path.isfile(os.path.join(lbl_dir, f)))

        # Supprimer les labels sans image correspondante
        for label_base in label_bases - image_bases:
            label_file = os.path.join(lbl_dir, label_base + '.txt')
            if os.path.exists(label_file):
                cptLabel += 1
                os.remove(label_file)
                print(f"Supprimé label orphelin: {label_file}")

        for image_base in image_bases - label_bases:
            for ext in ['.jpg', '.jpeg', '.png']:
                image_file = os.path.join(img_dir, image_base + ext)
                if os.path.exists(image_file):
                    cptImage += 1
                    os.remove(image_file)
                    print(f"Supprimé image orpheline: {image_file}")
    print(f"Total labels supprimés: {cptLabel}")
    print(f"Total images supprimées: {cptImage}") 


def count_elements_in_directory(directory_path):
    """
    Retourne le nombre total d'éléments (fichiers + sous-dossiers) dans le dossier spécifié.
    """
    if not os.path.exists(directory_path):
        print(f"Chemin invalide : {directory_path}")
        return 0

    print(len(os.listdir(directory_path)))

def move_files_with_only_class_0(labels_dir, images_dir, labels_mv_dir, images_mv_dir):
    """
    Déplace les images et labels dont les annotations contiennent uniquement la classe 0.
    """

    # Crée les dossiers de destination s'ils n'existent pas
    os.makedirs(labels_mv_dir, exist_ok=True)
    os.makedirs(images_mv_dir, exist_ok=True)

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue

        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        # Récupère les identifiants de classe
        class_ids = [line.split()[0] for line in lines]

        # Vérifie si toutes les classes sont '0'
        if class_ids and all(cls == '0' for cls in class_ids):
            # Déplace le label
            dest_label_path = os.path.join(labels_mv_dir, label_file)
            shutil.move(label_path, dest_label_path)

            # Déplace l'image correspondante (même nom, extensions possibles)
            base_name = os.path.splitext(label_file)[0]
            for ext in ['.jpg', '.jpeg', '.png']:
                image_file = base_name + ext
                image_path = os.path.join(images_dir, image_file)
                if os.path.exists(image_path):
                    dest_image_path = os.path.join(images_mv_dir, image_file)
                    shutil.move(image_path, dest_image_path)
                    break  # On a trouvé et déplacé l’image                


# Exemple d'utilisation :
# delete_clone1_files('chemin/vers/ton/dataset')



# Remplace 'datasets/forklift' par le chemin de ton dataset
# process_labels_and_images('datasets/forklift')
# replace_class('workers', '0', '1')
# replace_class('forklift_human_custom', '1', '0')
# replace_class('forklift_human_custom', '2', '1')
# clone_dataset('forklif_human_together', 'global_dataset')
# delete_clone1_files('global_dataset')
# sync_labels_and_images('global_dataset')
# count_elements_in_directory('../yolov5/datasets/global_dataset_workers/train/labels')
# move_files_with_only_class_0('global_dataset/valid/labels', 'global_dataset/valid/images', 'global_dataset2/valid/labels_0', 'global_dataset2/valid/images_0')
