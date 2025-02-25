import torch
import torch.nn as nn
from network import get_network
from utils import get_loader, Statistics, RSquaredStatistics, write_perf_to_json
from tqdm.auto import tqdm
import numpy as np
import time
import argparse
try:
    import neptune
except:
    neptune = None
import os

# Define R-squared Metric
def r_squared(y_true, y_pred, reduction:bool=True, joint:bool=False):
    '''
    If reduction=True, returns the average R-squared value over the feature.
    If reduction=False, returns a vector of R-squared values, one for each feature.
    '''
    with torch.no_grad():
        ss_res = ((y_true - y_pred) ** 2).sum(dim=0) # sum over batch index
        ss_tot = ((y_true - y_true.mean(dim=0, keepdim=True)) ** 2).sum(dim=0) # sum over batch index
        assert (ss_tot > 0).all(), "Variance of some y_true dimension is zero. R-squared is not defined when the variance = 0."
        if joint:
            return 1 - ss_res.sum() / ss_tot.sum()
        r2 = 1 - ss_res / ss_tot
        if reduction:
            return r2.mean()
        else:
            return r2

def train_loop(loader, model, loss_fn, optimizer, device, tau:float=0.5, separate_training:bool=False):
    model.train()
    model = model.to(device)
    circuit_loss_stat, physical_loss_stat, circuit_r2_stat, physical_r2_stat = Statistics.get_statistics(4, momentum=0.9)
    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        x, y, z = batch
        x, y, z = x.to(device), y.to(device), z.to(device)
        optimizer.zero_grad()
        if separate_training:
            circuit_pred, physical_pred = model(x, y=y)
        else:
            circuit_pred, physical_pred = model(x)
        loss_circuit = loss_fn(y, circuit_pred)
        loss_physical = loss_fn(z, physical_pred)
        loss = loss_circuit * tau + loss_physical
        r2_circuit = r_squared(y, circuit_pred)
        r2_physical = r_squared(z, physical_pred)
        circuit_loss_stat.update(loss_circuit.item())
        physical_loss_stat.update(loss_physical.item())
        circuit_r2_stat.update(r2_circuit.item())
        physical_r2_stat.update(r2_physical.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        pbar.set_description(f"Loss = {circuit_loss_stat.avg * tau + physical_loss_stat.avg:.3E}, Circuit R^2 = {circuit_r2_stat.avg:.3E}, Physical R^2 = {physical_r2_stat.avg:.3E}")
    return circuit_loss_stat.avg, physical_loss_stat.avg, circuit_r2_stat.avg, physical_r2_stat.avg

def test_loop(loader, model, loss_fn, device):
    model.eval()
    model = model.to(device)
    circuit_loss_stat, physical_loss_stat = Statistics.get_statistics(2)
    circuit_r2_stat, physical_r2_stat = RSquaredStatistics.get_statistics(2, feature_mean=0)
    with torch.no_grad():
        pbar = tqdm(loader, desc='Testing')
        for batch in pbar:
            x, y, z = batch
            x, y, z = x.to(device), y.to(device), z.to(device)
            circuit_pred, physical_pred = model(x)
            loss_circuit = loss_fn(y, circuit_pred)
            loss_physical = loss_fn(z, physical_pred)
            circuit_r2_stat.update(y, circuit_pred, len(x))
            physical_r2_stat.update(z, physical_pred, len(x))
            circuit_loss_stat.update(loss_circuit.item(), len(x))
            physical_loss_stat.update(loss_physical.item(), len(x))
            pbar.set_description(f"Circuit Loss = {circuit_loss_stat.avg:.3E}, Physical Loss: {physical_loss_stat.avg:.3E}, Circuit R^2 = {circuit_r2_stat.avg:.3E}, Physical R^2 = {physical_r2_stat.avg:.3E}")
    return circuit_loss_stat.avg, physical_loss_stat.avg, circuit_r2_stat.avg, physical_r2_stat.avg
    

if __name__ == '__main__':
    local_log = {}
    if neptune is not None:
        nep_log = neptune.init_run(
            project="use your own project name"
        )
        neptune_id = nep_log["sys/id"].fetch()
        local_log["sys/id"] = neptune_id
    else:
        nep_log = None

    parser = argparse.ArgumentParser()
    parser.add_argument("--netname", type=str, required=True)
    parser.add_argument("--data_type", type=str, default="NodeA_MetalOptionA_FreqA")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--save_root", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    
    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = get_network(args.netname).to(device)

    # Obtain the data loaders
    train_loader, test_loader, val_loader = get_loader(data_type=args.data_type, batch_size=4096)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

    # Using SMSE as the loss function
    loss_fn = nn.MSELoss()

    lr_milestones = [150, 200, 250]
    lr_decay_factor = 0.2
    lr_schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_milestones, gamma=lr_decay_factor)

    start_time = time.time()
    epochs = args.epochs
    tau = args.tau
    best_loss = np.inf
    best_cir_r2, best_phy_r2 = -np.inf, -np.inf

    save_root = os.path.join(args.save_root, args.data_type, args.netname, f"Epoch_{epochs}", f"tau_{tau}")
    os.makedirs(save_root, exist_ok=True)
    write_perf_to_json(local_log, save_root, filename="log.json")

    for epoch in range(epochs):
        train_cir_loss, train_phy_loss, train_cir_r2, train_phy_r2 = train_loop(train_loader, model, loss_fn, optimizer, device, tau=tau, separate_training=False)
        val_cir_loss, val_phy_loss, val_cir_r2, val_phy_r2 = test_loop(val_loader, model, loss_fn, device)
        val_loss = val_cir_loss * tau + val_phy_loss
        lr_schedular.step()
        if val_loss < best_loss:
            best_loss = val_loss
            best_cir_r2, best_phy_r2 = val_cir_r2, val_phy_r2
            torch.save(model.state_dict(), os.path.join(save_root, "model.pt"))
            print("Refreshed the best model.")
        print(f"Epoch {epoch+1}: Val circuit loss = {val_cir_loss:.3E}, Val physical loss = {val_phy_loss:.3E}, Val circuit R^2 = {val_cir_r2:.3E}, Val physical R^2 = {val_phy_r2:.3E}")
        if nep_log is not None:
            nep_log["results/val_loss"].append(val_cir_loss*tau + val_phy_loss)
            nep_log["results/train_loss"].append(train_cir_loss*tau + train_phy_loss)
            nep_log["results/val_cir_r2"].append(val_cir_r2)
            nep_log["results/val_phy_r2"].append(val_phy_r2)
            nep_log["results/train_cir_r2"].append(train_cir_r2)
            nep_log["results/train_phy_r2"].append(train_phy_r2)
            nep_log["results/lr"].append(optimizer.param_groups[0]["lr"])
        

    print(f"Training completed in {time.time() - start_time:.2f} seconds")

    # Load the best model and evaluate on the test set
    model.load_state_dict(torch.load(os.path.join(save_root, "model.pt")))
    test_cir_loss, test_phy_loss, test_cir_r2, test_phy_r2 = test_loop(test_loader, model, loss_fn, device)
    print(f"Test circuit loss = {test_cir_loss:.3E}, Test physical loss = {test_phy_loss:.3E}, Test circuit R^2 = {test_cir_r2:.3E}, Test physical R^2 = {test_phy_r2:.3E}")
    if nep_log is not None:
        nep_log["model"].upload(os.path.join(save_root, "model.pt"))
        nep_log["results/test_cir_loss"] = test_cir_loss
        nep_log["results/test_phy_loss"] = test_phy_loss
        nep_log["results/test_cir_r2"] = test_cir_r2
        nep_log["results/test_phy_r2"] = test_phy_r2
    local_log["results/test_cir_loss"] = test_cir_loss
    local_log["results/test_phy_loss"] = test_phy_loss
    local_log["results/test_cir_r2"] = test_cir_r2
    local_log["results/test_phy_r2"] = test_phy_r2
    write_perf_to_json(local_log, save_root, filename="log.json")