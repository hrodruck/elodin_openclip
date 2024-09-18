import torch
import torchvision.transforms as F
import torch.nn as nn
import clip
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

src_path = "src.jpg"
tgt_path = "tgt.jpg"

device = 'cuda'
mtcnn = MTCNN(image_size = 112, device = device).half()
resnet = InceptionResnetV1(pretrained = 'vggface2').to(device).eval().half()
resnet.classify = True
clip_model, _ = clip.load(name='ViT-B/16', device=device)

def get_id_embeds(images):
    images = images.to(device)
    faces =  mtcnn(images).to(device)/255
    return resnet(faces), faces

def compute_similarity(text, image_vector):
        target_tokens = clip.tokenize(text).to(device)
        text_vector_tgt = clip_model.encode_text(target_tokens)
        text_vector_tgt = torch.cat (image_vector.shape[0]*[text_vector_tgt])
        return nn.CosineSimilarity()(image_vector, text_vector_tgt)

def compare_face_face(source_path, target_path):
        print ('starting face id....')
        
        with torch.no_grad():
            source_img = Image.open(source_path).convert('RGB')
            source_img = F.ToTensor()(source_img).to(device).to(dtype = torch.half).unsqueeze(0)
            source_embeds, faces = get_id_embeds(source_img)

            target_img = Image.open(target_path).convert('RGB')
            target_img = F.ToTensor()(target_img).to(device).to(dtype = torch.half).unsqueeze(0)
            target_embeds, _ = get_id_embeds(target_img)
        
        source_img = nn.functional.interpolate(source_img, size=(224, 224), mode = 'bicubic', align_corners=True)
        source_vector = clip_model.encode_image(source_img)
        reg_similarity = compute_similarity("A photo of a woman", source_vector)
        
        
        faces_failed = torch.amax(faces, dim = (1, 2, 3)) < 1e-3
        source_embeds = source_embeds[~faces_failed]
        
        if torch.numel(source_embeds)>0:
            target_emb_batch = torch.cat(source_embeds.shape[0]*[target_embeds])
            score = nn.CosineSimilarity()(target_emb_batch, source_embeds)
            loss = score.mean()
            print(f'{loss=}')
        else: #faces not detected
            dummy_loss = (1e-5*reg_similarity).mean() 
            print (f'{dummy_loss=}')
            loss = dummy_loss
        return loss

def compare_tensor_face(source_tensor, target_path):
        print ('starting face id....')
        
        with torch.no_grad():
            source_img = source_tensor.to(device).to(dtype = torch.half).unsqueeze(0)
            source_embeds, faces = get_id_embeds(source_img)

            target_img = Image.open(target_path).convert('RGB')
            target_img = F.ToTensor()(target_img).to(device).to(dtype = torch.half).unsqueeze(0)
            target_embeds, _ = get_id_embeds(target_img)
        
        source_img = nn.functional.interpolate(source_img, size=(224, 224), mode = 'bicubic', align_corners=True)
        source_vector = clip_model.encode_image(source_img)
        reg_similarity = compute_similarity("A photo of a woman", source_vector)
        
        
        faces_failed = torch.amax(faces, dim = (1, 2, 3)) < 1e-3
        source_embeds = source_embeds[~faces_failed]
        
        if torch.numel(source_embeds)>0:
            target_emb_batch = torch.cat(source_embeds.shape[0]*[target_embeds])
            score = nn.CosineSimilarity()(target_emb_batch, source_embeds)
            loss = score.mean()
            print(f'{loss=}')
        else: #faces not detected
            dummy_loss = (1e-5*reg_similarity).mean() 
            print (f'{dummy_loss=}')
            loss = dummy_loss
        return loss

if __name__ == "__main__":
    compare_face_face(src_path, tgt_path)