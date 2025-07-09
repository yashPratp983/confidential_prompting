from __future__ import annotations

import fire
import os
import multiprocessing
import math
import torch

import datetime

import pickle

import numpy as np
import torch
import fire
import torch.distributed as dist
from transformers import AutoTokenizer, AutoConfig
from multiprocessing import shared_memory
from attention import AttentionBuffer
from llama import LlamaForCausalLM, softmax
from prompt import PublicMeta, PrivateMeta, Replacement
import torch
import torch.multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import os
import math  # Added missing import

class Streamprinter:

    def __init__(self):
        self.prev = 0

    def print(self, text):
        text = text.strip()
        now = len(text) - 1
        if now > self.prev:
            print(text[self.prev:now], end="", flush=True)
            self.prev = now
        # #print(" ".join(text[self.prev:]), flush=True)


class SharedMemoryManager:
    """Manages shared memory for tensors between processes"""
    def __init__(self,num_layers, num_users, gamma, num_heads, head_dim, dtype=torch.float,q_shms=[], o_shms=[],sync_shm=None):
        self.num_users = num_users
        self.gamma = gamma
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.num_layers=num_layers
        # Calculate sizes for tensors
        self.q_tensor_size = (gamma, num_heads, 1, head_dim)
        self.o_tensor_size = (gamma, num_heads, 1, head_dim + 2)
        
        # Calculate bytes needed
        self.q_bytes = np.prod(self.q_tensor_size) * (2 if dtype == torch.float else 4)
        self.o_bytes = np.prod(self.o_tensor_size) * (2 if dtype == torch.float else 4)
        
        # Create shared memory blocks for each user
        self.q_shms = q_shms
        self.o_shms = o_shms
        if len(self.q_shms)==0:
            #print("Creating shared memory for Q tensors")
            for _ in range(num_users):
                q_shm = shared_memory.SharedMemory(create=True, size=100000000)
                o_shm = shared_memory.SharedMemory(create=True, size=100000000)
                self.q_shms.append(q_shm)
                self.o_shms.append(o_shm)
            
        # Create indicators for synchronization - adding a termination flag
        if sync_shm is not None:
            self.sync_shm = sync_shm
            self.sync_array = np.ndarray((num_users, 3), dtype=np.uint8, buffer=self.sync_shm.buf)
        else:
            self.sync_shm = shared_memory.SharedMemory(create=True, size=num_users * 3)
            self.sync_array = np.ndarray((num_users, 3), dtype=np.uint8, buffer=self.sync_shm.buf)
            self.sync_array.fill(0)  # Initialize all flags to 0
        # sync_array[:, 0] is q_ready flag
        # sync_array[:, 1] is o_ready flag
        # sync_array[:, 2] is termination flag
        
    def get_q_tensor_master(self, user_id, device=None):
        """Get a tensor view of the Q tensor for the master process"""
        array = np.ndarray(self.q_tensor_size, 
                          dtype=np.float32 if self.dtype == torch.float else np.float32, 
                          buffer=self.q_shms[user_id].buf)
        tensor = torch.from_numpy(array).clone()
        if device:
            tensor = tensor.to(device)
        return tensor
    
    def get_o_tensor_master(self, user_id, device=None):
        """Get a tensor view of the O tensor for the master process"""
        o_tensor_size = (self.num_layers, *self.o_tensor_size)  # Add num_layers as first dimension
        
        array = np.ndarray(
            o_tensor_size,
            dtype=np.float16 if self.dtype == torch.float16 else np.float32,
            buffer=self.o_shms[user_id].buf
        )
        tensor = torch.from_numpy(array)
        if device:
            tensor = tensor.to(device)
        return tensor
    
    def get_q_tensor_worker(self, user_id, device=None):
        """Get a tensor view of the Q tensor for the worker process"""
        array = np.ndarray(self.q_tensor_size, 
                          dtype=np.float32 if self.dtype == torch.float else np.float32, 
                          buffer=self.q_shms[user_id].buf)
        tensor = torch.from_numpy(array)
        if device:
            tensor = tensor.to(device)
        return tensor
    
    def get_o_tensor_worker(self, user_id, device=None):
        """Get O tensor view for worker process"""
        # Shape: (num_layers, n, h, q_len, output_features)
        # where output_features = 2 + attention_dim
        o_tensor_size = (self.num_layers, *self.o_tensor_size)  # Add num_layers as first dimension
        
        array = np.ndarray(
            o_tensor_size,
            dtype=np.float16 if self.dtype == torch.float16 else np.float32,
            buffer=self.o_shms[user_id].buf
        )
        tensor = torch.from_numpy(array)
        if device:
            tensor = tensor.to(device)
        return tensor
    
    def signal_q_ready(self, user_id):
        """Signal that Q tensor is ready for processing"""
        self.sync_array[user_id, 0] = 1
        #print("signal_q_called")
        
    def signal_o_ready(self, user_id):
        """Signal that O tensor is ready for processing"""
        self.sync_array[user_id, 1] = 1
        #print("signal_o_called")
    
    def signal_termination(self):
        """Signal all workers to terminate"""
        self.sync_array[:, 2] = 1
        
    def check_termination(self, user_id):
        """Check if termination has been signaled"""
        return self.sync_array[user_id, 2] == 1
        
    def wait_for_q_ready(self, user_id):
        """Wait until Q tensor is ready"""
        while self.sync_array[user_id, 0] != 1:
            # Check for termination signal
            if self.sync_array[user_id, 2] == 1:
                return False
            # Add a small sleep to reduce CPU usage
            # import time
            # time.sleep(0.001)
        
        # Reset for next round
        return True
    
    def make_wait_q(self,user_id):
        self.sync_array[user_id, 0] = 0  
        return True
        
    def wait_for_o_ready(self, user_id):
        """Wait until O tensor is ready"""
        while self.sync_array[user_id, 1] != 1:
            # Check for termination signal
            if self.sync_array[user_id, 2] == 1:
                return False
            # Add a small sleep to reduce CPU usage
            # import time
            # time.sleep(0.001)
            
        # Reset for next round
        self.sync_array[user_id, 1] = 0  
        return True
    
    def cleanup(self):
        """Clean up shared memory"""
        for shm in self.q_shms:
            shm.close()
            shm.unlink()
        for shm in self.o_shms:
            shm.close()
            shm.unlink()
        self.sync_shm.close()
        self.sync_shm.unlink()



class AttentionVaultShared:
    """Replacement for AttentionVault using shared memory"""
    
    def __init__(self, buffer: object, gamma: int, attention_mask: torch.Tensor, 
                 num_group: int, user_id: int, shared_mem_manager: SharedMemoryManager,
                 device: str = 'cuda:0'):
        self.kv_buffer = buffer
        self.num_layers = buffer.num_layers
        self.head_dim = buffer.head_dim
        self.num_heads = buffer.num_heads
        self.num_group = num_group
        self.gamma = gamma
        self.device = device
        self.user_id = user_id
        self.shared_mem = shared_mem_manager
        
        # Get tensor views from shared memory
        self.q_buffer = self.shared_mem.get_q_tensor_worker(user_id, device)
        self.o_buffer = self.shared_mem.get_o_tensor_worker(user_id, device)
        
        # Invert mask if provided
        if attention_mask is not None:
            inverted_mask = 1.0 - attention_mask.float()
            self.attention_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), 
                                                          torch.finfo(buffer.dtype).min).to(buffer.dtype)
        else:
            self.attention_mask = None
    
    @torch.inference_mode()
    def serve(self):
        """Main service loop for processing attention computations"""
        # print(f"Worker {self.user_id}: Starting service for {self.num_layers} layers")
        
        for layer_idx in range(self.num_layers):
            # Wait for Q state to be ready
            if not self.shared_mem.wait_for_q_ready(self.user_id):
                # print(f"Worker {self.user_id}: Termination signal received at layer {layer_idx}")
                return False
            
            # Compute private attention
            q_new = self.q_buffer.clone()  # Work with a copy
            k_pvt, v_pvt = self.kv_buffer.cache(layer_idx, self.num_group)
            
            # Compute attention scores: (n, h, q_len, d) @ (n, h, kv_seq_len, d)^T
            score_pvt = torch.matmul(q_new, k_pvt.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            # Apply attention mask if present
            if self.attention_mask is not None:
                score_pvt = score_pvt + self.attention_mask.unsqueeze(1).unsqueeze(1)
            
            # Softmax computation (simplified version)
            max_pvt = torch.max(score_pvt, dim=-1, keepdim=True)[0]
            score_exp = torch.exp(score_pvt - max_pvt)
            sum_pvt = torch.sum(score_exp, dim=-1, keepdim=True)
            score_pvt = score_exp / sum_pvt
            
            # Compute attention output
            attn_pvt = torch.matmul(score_pvt, v_pvt)
            
            # Store results in output buffer for this layer: [max, sum, attention_output]
            self.o_buffer[layer_idx, :, :, :, 0:1] = max_pvt
            self.o_buffer[layer_idx, :, :, :, 1:2] = sum_pvt  
            self.o_buffer[layer_idx, :, :, :, 2:] = attn_pvt
            
            # print(f"Worker {self.user_id}: Completed layer {layer_idx}")
        self.shared_mem.make_wait_q(self.user_id)
        # Signal that all outputs are ready after processing all layers
        self.shared_mem.signal_o_ready(self.user_id)

        # print(f"Worker {self.user_id}: All {self.num_layers} layers completed, output ready")
        
        return True


@torch.inference_mode()
def init_master_shared(
    states_dir:str,
    model_path:str, 
    device:str,
    num_users:int=1,
    capacity:int = 1024 * 2,
    o_shared_mem=None,
    q_shared_mem=None,
    sync_shm=None,
    print_idx:int=0,
    gamma:int = 1,
    num_heads:int = 1,
    head_dim:int = 1
    ):
    """Master process implementation using shared memory"""
    
    # print("Master started.")
    
    with open(os.path.join(states_dir, "public.meta"), "rb") as f:
        meta = pickle.load(f)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float,
        device_map=device)
    shared_mem= SharedMemoryManager(
        num_layers=model.config.num_hidden_layers,
        num_users=num_users,
        gamma=gamma,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=torch.float,
        q_shms=q_shared_mem,
        o_shms=o_shared_mem,
        sync_shm=sync_shm
    )


    
    

    
    # Set the shared memory manager in all attention layers
    for layer in model.model.layers:
        layer.self_attn.set_shared_memory(shared_mem, device)

    buffer = AttentionBuffer(num_batch=meta.gamma,
                             capacity=capacity,
                             num_layers=model.config.num_hidden_layers,
                             num_heads=model.config.num_key_value_heads,
                             head_dim=model.config.hidden_size // model.config.num_attention_heads,
                             dtype=torch.float,
                             device=device)

    buffer.load(os.path.join(states_dir, meta.path))

    token_ids = meta.initial_token_ids
    position_ids = [meta.pos_offset] * meta.gamma
    buffer_sink_ids = buffer.allocate(1)

    output_ids = [[] for _ in range(meta.gamma)]
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]

    # Set a maximum number of iterations to prevent infinite loops
    max_iterations = 100
    iteration_count = 0
    
    # Stream #printer to display output (assuming Stream#printer is imported)
    try:
        printer = Streamprinter()
    except ImportError:
        # Fallback simple #printer if Stream#printer is not available
        class Simpleprinter:
            def __init__(self):
                self.text = ""
            def print(self, text):
                print("\r" + text, end="", flush=True)
        printer = Simpleprinter()
    
    try:
        while iteration_count < max_iterations:
            iteration_count += 1
            
            logits = model(
                input_ids=torch.as_tensor(token_ids, device=device).unsqueeze(-1),
                position_ids=torch.as_tensor(position_ids, device=device).unsqueeze(-1),
                buffer=buffer,
                buffer_sink_ids=buffer_sink_ids,
                confidential=True,
                num_users=num_users
            )
            
            # Sample from logits
            last_token_logits = logits[:, -1, :]
            new_token = torch.argmax(last_token_logits, dim=-1).tolist()
            token_ids = new_token

            buffer_sink_ids = buffer.allocate(1)

            for i in range(meta.gamma):
                output_ids[i].append(new_token[i])
                
            position_ids = [len(output_ids[0]) + meta.pos_offset] * meta.gamma

            printer.print(tokenizer.decode(output_ids[print_idx]))

            if new_token[0] in stop_token_ids:
                print("\nGeneration complete - stopping token generated.")
                break
                
            # Add a condition to break if reached a reasonable number of tokens
            # if len(output_ids[0]) >= 50:  # Adjust as needed
            #     print("\nGeneration reached target length.")
            #     break
                
    except Exception as e:
        print(f"\nError in master process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Signal all workers to terminate
        #print("\nMaster signaling termination to workers")
        shared_mem.signal_termination()
        
        try:
            # Try to #print logs if available
            import logger
            result = logger.read()
            processed = logger.process_data(result)
            
            #print("====================")
            #print(processed)
            #print("====================")
        except Exception as log_error:
            print(f"Failed to process logs: {log_error}")


def init_worker_shared(
    states_dir:str,
    model_path:str,
    device:str,
    user_id:int = 0,
    o_shared_mem=None,
    q_shared_mem=None,
    sync_shm=None,
    disable_multiplexing:bool = False,
    num_users:int = 1,  
    gamma:int = 1,
    num_heads:int = 1,
    head_dim:int = 1
    ):
    """Worker process implementation using shared memory"""
    
    #print(f"Worker {user_id} started.")
    
    try:

        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        shared_mem= SharedMemoryManager(
        num_layers=config.num_hidden_layers,
        num_users=num_users,
        gamma=gamma,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=torch.float,
        q_shms=q_shared_mem,
        o_shms=o_shared_mem,
        sync_shm=sync_shm
    )


            
        with open(os.path.join(states_dir, "private.meta"), "rb") as f:
            meta = pickle.load(f)

        
        
        # The buffer does not grow, so we can allocate the right size from the start
        buffer_private = AttentionBuffer(num_batch=1,
                                        capacity=meta.len_private,
                                        num_layers=config.num_hidden_layers,
                                        num_heads=config.num_key_value_heads,
                                        head_dim=config.hidden_size // config.num_attention_heads,
                                        dtype=torch.float,
                                        device=device)
        
        buffer_private.load(os.path.join(states_dir, meta.path))

        # Create virtual prompts
        virtual_prompt_buffer_ids = [[] for _ in range(meta.gamma)]

        for reps in meta.replacements.values():
            for i in range(meta.gamma):
                j = i % len(reps)
                rel_ids = reps[j].buffer_ids
                virtual_prompt_buffer_ids[i].extend(rel_ids)

        common_mask = np.zeros(meta.len_private, dtype=np.bool_)
        offset = 0
        for mask_type, mask_size in meta.mask_info:
            if mask_type == 0:
                common_mask[offset:offset + mask_size] = True
            offset += mask_size

        mask = np.zeros((meta.gamma, meta.len_private), dtype=np.bool_)
        mask[:, :] = common_mask

        for i in range(meta.gamma):
            mask[i, virtual_prompt_buffer_ids[i]] = True

        if disable_multiplexing:
            new_cap = max(len(mask[i].nonzero()[0]) for i in range(meta.gamma))
            
            buffer_private2 = AttentionBuffer(num_batch=meta.gamma,
                                        capacity=new_cap,
                                        num_layers=config.num_hidden_layers,
                                        num_heads=config.num_key_value_heads,
                                        head_dim=config.hidden_size // config.num_attention_heads,
                                        dtype=torch.float,
                                        device=device)
            
            for i in range(meta.gamma):
                indices = mask[i].nonzero()[0]
                
                for layer_idx in range(buffer_private.num_layers):
                    buffer_private2.k[layer_idx][i, :, :len(indices), :].copy_(buffer_private.k[layer_idx][0, :, indices, :])
                    buffer_private2.v[layer_idx][i, :, :len(indices), :].copy_(buffer_private.v[layer_idx][0, :, indices, :])
            
            buffer_private2.allocate(new_cap)
            
            buffer_private.clear()
            del buffer_private
            buffer_private = buffer_private2
            
            mask = None
            
        if mask is not None:
            mask = torch.as_tensor(mask, device=device)

        # Create attention vault using shared memory
        vault = AttentionVaultShared(
            buffer_private, 
            meta.gamma, 
            mask, 
            num_group=config.num_attention_heads // config.num_key_value_heads, 
            user_id=user_id,
            shared_mem_manager=shared_mem,
            device=device
        )

        # #print memory consumption in MB
        #print(f"Worker {user_id} memory consumption: {buffer_private.memory_consumption() / 1024 / 1024:.2f} MB")

        # Main service loop with termination check
        while not shared_mem.check_termination(user_id):
            vault.serve()
                
                
    except Exception as e:
        print(f"Error in worker {user_id}: {e}")
        import traceback
        traceback.print_exc()
    
    #print(f"Worker {user_id} shutting down")


def main_shared_memory(model="meta-llama/Llama-3.2-3B-Instruct",
         device="cuda:0",
         states_dir="./states",
         num_users=1,
         timeout_sec=5,
         max_num_tokens=2048,
         standalone_master=False,
         standalone_worker=False,
         user_id=0,
         print_idx=0,
         disable_multiplexing=False
         ):
    """Main function using shared memory for communication instead of distributed"""
    import torch
    from transformers import AutoConfig
    
    # Load public and private metadata
    with open(os.path.join(states_dir, "public.meta"), "rb") as f:
        public_meta = pickle.load(f)
    
    # Load model config to get dimensions
    config = AutoConfig.from_pretrained(model)
    head_dim = config.hidden_size // config.num_attention_heads
    
    # Create shared memory manager
    shared_mem = SharedMemoryManager(
        num_layers=config.num_hidden_layers,
        num_users=num_users,
        gamma=public_meta.gamma,
        num_heads=config.num_attention_heads,
        head_dim=head_dim,
        dtype=torch.float  # Use float16 for shared memory to reduce size
    )
    
    try:
        if standalone_master:
            # Initialize and run master process only
            init_master_shared(
                states_dir=states_dir,
                model_path=model,
                device=device,
                capacity=max_num_tokens,
                o_shared_mem=shared_mem.o_shms,
                q_shared_mem=shared_mem.q_shms,
                sync_shm=shared_mem.sync_shm,
                print_idx=print_idx,
                num_users=num_users,
                gamma=public_meta.gamma,
                num_heads=config.num_attention_heads,
                head_dim=head_dim,
            )
            return
        
        if standalone_worker:
            # Initialize and run a single worker only
            init_worker_shared(
                states_dir=states_dir,
                model_path=model,
                device=device,
                user_id=user_id,
                sync_shm=shared_mem.sync_shm,
                o_shared_mem=shared_mem.o_shms,
                q_shared_mem=shared_mem.q_shms,
                disable_multiplexing=disable_multiplexing,
                num_users=num_users,
                gamma=public_meta.gamma,
                num_heads=config.num_attention_heads,
                head_dim=head_dim,
            )
            return
        
        # Start all processes
        processes = []
        master_process = mp.Process(target=init_master_shared, kwargs={
            'states_dir': states_dir,
            'model_path': model,
            'device': device,
            'num_users': num_users,
            'capacity': max_num_tokens,
            'o_shared_mem':shared_mem.o_shms,
            'q_shared_mem':shared_mem.q_shms,
            'sync_shm': shared_mem.sync_shm,
            'print_idx': print_idx,
            'num_users': num_users,
            'gamma': public_meta.gamma,
            'num_heads': config.num_attention_heads,
            'head_dim': head_dim
        })
        master_process.start()
        # Start worker processes
        for i in range(num_users):
            p = mp.Process(target=init_worker_shared, kwargs={
                'states_dir': states_dir,
                'model_path': model,
                'device': device,
                'user_id': i,
                'o_shared_mem':shared_mem.o_shms,
                'q_shared_mem':shared_mem.q_shms,
                'disable_multiplexing': disable_multiplexing,
                'sync_shm': shared_mem.sync_shm,
                'num_users': num_users,
                'gamma': public_meta.gamma,
                'num_heads': config.num_attention_heads,
                'head_dim': head_dim

            })
            p.start()
            processes.append(p)
        
        # Add a small delay to ensure workers are initialized
        import time
        time.sleep(2)
        
        # Start master process
        
        processes.append(master_process)
        # Wait for master to complete
        
        # if master_process.is_alive():
        #     #print("Master process timed out, terminating...")
        #     shared_mem.signal_termination()
        #     master_process.terminate()
        #     master_process.join(1)
        
        # Wait for worker processes to terminate
        for p in processes:
            p.join()
            if p.is_alive():
                p.terminate()
                p.join(1)
                
    except Exception as e:
        print(f"Error in main process: {e}")
    finally:
        # Clean up shared memory
        shared_mem.cleanup()
        #print("Shared memory cleaned up")


if __name__ == "__main__":
    import sys
    torch.multiprocessing.set_start_method('spawn')
    
    if len(sys.argv) > 1:
        import fire
        fire.Fire(main_shared_memory)
    else:
        # Default parameters for testing
        main_shared_memory()