from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader
        
        # Model configurations.
        self.c_dim = config.c_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = 'CelebA'
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    
    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None, mode='train'):
        """Generate target domain labels for debugging and testing."""
        c_trg_list = []
        data_loader = self.celeba_loader
        c_trg = c_org.clone()
        '''
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
		
		# Fetch real images and labels.
        try:
            x_real, label_org = next(data_iter)
                #print(type(x_real),x_real.size())         # <class 'torch.Tensor'> torch.Size([16, 3, 128, 128])
                #print(type(label_org),label_org.size())   # <class 'torch.Tensor'> torch.Size([16, 1]) 
        except:
            data_iter = iter(data_loader)
            x_real, label_org = next(data_iter)
		
        x_real = x_real.to(self.device)
        '''
        
        if mode=='test':
            for i, (x_real, c_org) in enumerate(data_loader):    
                #print(x_real.size(),x_real)
                #print(c_org.size(),c_org)
               
                
				# Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                #c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
                out_src, out_cls = self.D(x_real)
                if out_cls>0:
                    c_trg[:, 0]=0
                else:
                    c_trg[:, 0]=1
                #print('Out_cls : ',out_cls,' i= ',i+1)
                c_trg_list.append(c_trg.to(self.device))     
            return c_trg_list
        
        c_trg_list = []
        i=0
        c_trg = c_org.clone()
        c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
        c_trg_list.append(c_trg.to(self.device))
        #print('c_org',c_org)												Reverses the original labels, for Male
        #print('c_trg_list',c_trg_list)
		
        return c_trg_list
	
    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        
    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        data_loader = self.celeba_loader
       
        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
		
        D1=[]
        D2=[]
        D3=[]
        D4=[]
        G1=[]
        G2=[]
        G3=[]
        epochs=[]
		
        for i in range(start_iters, (self.num_iters)):
			
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
                #print(type(x_real),x_real.size())         # <class 'torch.Tensor'> torch.Size([16, 3, 128, 128])
                #print(type(label_org),label_org.size())   # <class 'torch.Tensor'> torch.Size([16, 1]) 
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            #print('asdfs')
            # Generate target domain labels randomly.
            #rand_idx = torch.randperm(label_org.size(0))
            #label_trg = label_org[rand_idx]
			
            label_trg = label_org.clone()
            label_trg[:, 0] = (label_org[:, 0] == 0)
			
            #print(label_trg)
            c_org = label_org.clone()				#Actual labels from list_attr_celeb.txt
            c_trg = label_trg.clone()				#Batch size(16) generated random labels
            
            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            #print(type(out_src),out_src.size())    #<class 'torch.Tensor'> torch.Size([16, 1, 2, 2])
            #print(type(out_cls),out_cls.size())    # <class 'torch.Tensor'> torch.Size([16, 1])
			
			
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(xxxxxxxxxxx)    			# pRINT THIS AND CHECK

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            #print('out_src,out_src.size()0',out_src,out_src.size())
            #print('x_hat,x_hat.size()',x_hat,x_hat.size())
            d_loss_gp = self.gradient_penalty(out_src, x_hat)
            #print('d_loss_gp' ,d_loss_gp)
			
            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
			
            #print('D1: ',D1)
            #print('size',len(D1))
            #print(i)
            #D1.append(d_loss_real)
            #D2.append(d_loss_fake)
            #D3.append(d_loss_cls)
            #D4.append(d_loss_gp)
            
			# =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
				
                #print('G/LOSS : ',g_loss_rec.item())
                #G1.append(g_loss_fake)
                #G2.append(g_loss_rec)
                #G3.append(g_loss_cls)
				
            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
					
                    value="{:.4f}".format(value)
                    #print(value)
                    if tag=='D/loss_real':
                        D1.append(value)
                        #print (value)
                    if tag=='D/loss_fake':
                        D2.append(value)
                    if tag=='D/loss_cls':
                        D3.append(value)
                    if tag=='D/loss_gp':
                        D4.append(value)
					
                    if tag=='G/loss_fake':
                        G1.append(value)
                    if tag=='G/loss_rec':
                        G2.append(value)
                    if tag=='G/loss_cls':
                        G3.append(value)
					    
                print(log)
                #print('D1: ',D1)
                #print('size',len(D1))
                #print(i)
                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]	
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.G(x_fixed, c_fixed))
                        #print(len(x_fake_list),'asdf')
                    
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
				
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))
				
                for j in range(0,len(D1)):
                    epochs.append(j)
				
                
                #print(epochs)
                #print(len(epochs))
                
                
                
				
                D1=[float(i) for i in D1]
                D2=[float(i) for i in D2]
                D3=[float(i) for i in D3]
                D4=[float(i) for i in D4]

                G1=[float(i) for i in G1]
                G2=[float(i) for i in G2]
                G3=[float(i) for i in G3]
                epochs=[float(i) for i in epochs]
				
                df=pd.DataFrame({'D1': D1,'D2': D2,'D3': D3,'D4': D4,'G1': G1,'G2': G2,'G3': G3})
             
				
                df.to_csv('D:\Graphs.csv',mode='a', header=False)
                #df.to_csv('D:\Graphs.csv')
				
                '''
                ax=sns.tsplot(D1, epochs)
				#plt.plot()
                ax.figure.savefig(r'D:\D1.png', format='png', encoding='utf-8')
				
                ax=sns.tsplot(D2, epochs)
				#plt.plot(D2, epochs)
                ax.figure.savefig(r'D:\D2.png', format='png', encoding='utf-8')
			
                ax=sns.tsplot(D3, epochs)
				#plt.plot(D3, epochs)
                ax.figure.savefig(r'D:\D3.png', format='png', encoding='utf-8')
				
                ax=sns.tsplot(D4, epochs)
				#plt.plot(D4, epochs)
                ax.figure.savefig(r'D:\D4.png', format='png', encoding='utf-8')
				
                ax=sns.tsplot(G1, epochs)
				#plt.plot(G1, epochs)
                ax.figure.savefig(r'D:\G1.png', format='png', encoding='utf-8')
				
                ax=sns.tsplot(G2, epochs)
				#plt.plot(G2, epochs)
                ax.figure.savefig(r'D:\G2.png', format='png', encoding='utf-8')
				
                ax=sns.tsplot(G3, epochs)
				#plt.plot(G3, epochs)
                ax.figure.savefig(r'D:\G3.png', format='png', encoding='utf-8')
				
                ax=sns.tsplot(D2, G1)
				#plt.plot(D2, G1)
                ax.figure.savefig(r'D:\D2_G1.png', format='png', encoding='utf-8')
				
                ax=sns.tsplot(D3, G3)
				#plt.plot(D3, G3)
                ax.figure.savefig(r'D:\D3_G3.png', format='png', encoding='utf-8')
                '''
                
                D1=[]
                D2=[]
                D3=[]
                D4=[]
                G1=[]
                G2=[]
                G3=[]
                epochs=[]
		

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        
        data_loader = self.celeba_loader
        extracted=[]
        z_min=0
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                #print(x_real.size(),x_real)
                #print(c_org.size(),c_org)
                batch_loss=10
                batch_g={}
                x_real = x_real.to(self.device)
                z_max=0
                z=0
                #print(c_org)
                out_src, out_cls = self.D(x_real)
                if out_cls>0:
                    c=1
                else:
                    c=0
				
                c_trg = c_org.clone()
                c_trg[0,0] =c 
				#c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
				#c_trg_list.append(c_trg.to(self.device))
                c_org=c_trg=c_trg.to(self.device)
                #print(c_org)
				# Prepare input images and target domain labels.
                
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset,self.selected_attrs, mode='train')
                #print(c_trg_list)
                #out_src, out_cls = self.D(x_real)
                #if out_cls>0:
                #    c_trg_list=[0]
                #else:
                #    c_trg_list=[1]
                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))

                    out_src, out_cls = self.D(x_real)
                    #print(type(out_src),out_src.size())    #<class 'torch.Tensor'> torch.Size([16, 1, 2, 2])
                    #print(out_src)
                    #print(type(out_cls),out_cls.size())    # <class 'torch.Tensor'> torch.Size([16, 1])
                    #print('Out_cls : ',out_cls,' i= ',i+1)
                    x_reconst = self.G(x_real, c_trg)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))
                    #g_loss_rec_max,max = torch.max(torch.abs(x_real - x_reconst))
					
                    #print('rahul ', g_loss_rec)
                    batch_g['G/loss_rec'] = g_loss_rec.item()
						
                    for tag, value in batch_g.items():
                        value="{:.4f}".format(value)
                        value=float(value)
                        extracted.append(value)
                        #print(value)
                        if batch_loss>value:
                            batch_loss=value
                            z_min=i	
                            #print('batch_loss : ',value)
                        z=z+1
						
						
						
                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))
			
            '''
			extracted= [x_real]
            print(len(extracted))
            extracted.append(self.G(x_fixed, c_fixed_list[z_min]))
                
			x_concat = torch.cat(extracted, dim=3)
            sample_path = os.path.join(self.sample_dir, '{}-bestimages.jpg'.format(i+1))
            save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
            '''
            #print(z_min)
            top_2_idx = np.argsort(extracted)[-5:]
            top_2_values = [extracted[i] for i in top_2_idx]
            #print(top_2_idx)
            #print(top_2_values)
			
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):
                if i not in top_2_idx:
                    continue
                #print(x_real.size(),x_real)
                #print(c_org.size(),c_org)
                batch_loss=10
                batch_g={}
                z_max=0
                z=0
                
				# Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
				
                z_max=0
                z=0
                #print(c_org)
                out_src, out_cls = self.D(x_real)
                if out_cls>0:
                    c=1
                else:
                    c=0
				
                c_trg = c_org.clone()
                c_trg[0,0] =c 
				#c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
				#c_trg_list.append(c_trg.to(self.device))
                c_org=c_trg=c_trg.to(self.device)
				
				
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))

					
                    x_reconst = self.G(x_real, c_trg)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))
                    #g_loss_rec_max,max = torch.max(torch.abs(x_real - x_reconst))
					
                    #print('rahul ', g_loss_rec)
                    batch_g['G/loss_rec'] = g_loss_rec.item()
						
                    for tag, value in batch_g.items():
                        value="{:.4f}".format(value)
                        value=float(value)
                        extracted.append(value)
                        #print(value)
                        if batch_loss>value:
                            batch_loss=value
                            z_min=i	
                            #print('batch_loss : ',value)
                        z=z+1
						
						
						
                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join('stargan_celeba1/results1/extracted/best', '{}-extracted-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved best 5 real and fake images into {}...'.format(result_path))
			
            top_2_idx = np.argsort(extracted)[:5]
            top_2_values = [extracted[i] for i in top_2_idx]
            #print(top_2_idx)
            #print(top_2_values)
			
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):
                if i not in top_2_idx:
                    continue
                #print(x_real.size(),x_real)
                #print(c_org.size(),c_org)
                batch_loss=10
                batch_g={}
                z_max=0
                z=0
                
				# Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
				
                z_max=0
                z=0
                #print(c_org)
                out_src, out_cls = self.D(x_real)
                if out_cls>0:
                    c=1
                else:
                    c=0
				
                c_trg = c_org.clone()
                c_trg[0,0] =c 
				#c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
				#c_trg_list.append(c_trg.to(self.device))
                c_org=c_trg=c_trg.to(self.device)
				
				
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))

					
                    x_reconst = self.G(x_real, c_trg)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))
                    #g_loss_rec_max,max = torch.max(torch.abs(x_real - x_reconst))
					
                    #print('rahul ', g_loss_rec)
                    batch_g['G/loss_rec'] = g_loss_rec.item()
						
                    for tag, value in batch_g.items():
                        value="{:.4f}".format(value)
                        value=float(value)
                        extracted.append(value)
                        #print(value)
                        if batch_loss>value:
                            batch_loss=value
                            z_min=i	
                            #print('batch_loss : ',value)
                        z=z+1
						
						
						
                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join('stargan_celeba1/results1/extracted/worst', '{}-extracted-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved worst 5 real and fake images into {}...'.format(result_path))