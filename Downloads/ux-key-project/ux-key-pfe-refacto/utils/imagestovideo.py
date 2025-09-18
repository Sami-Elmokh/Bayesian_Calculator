import cv2
import os

def images_to_video(input_folder, output_video, fps):
    image_files = sorted(os.listdir(input_folder), key=lambda x: int(x.split('_')[1].split('.')[0]))

    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    height, width, _ = first_image.shape
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)
    video_writer.release()
    #cv2.destroyAllWindows()


images_to_video(input_folder = 'output\\gamma_images', output_video = 'output\\gamma_video\\output_video.mp4', fps=15) #note: fps: number of images per second
