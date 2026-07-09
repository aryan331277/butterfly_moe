def exp2_accuracy_memory_and_diversity(train_loader, val_loader, vocab_size, device,
                                        d_model, num_layers, top_k, num_epochs):

 
    expert_counts = [16, 32, 64]
    results = []
    diversity_scores = []
 
    n_diversity_plots = len(expert_counts)
    fig_div, axes_div = plt.subplots(1, n_diversity_plots, figsize=(5 * n_diversity_plots, 5))
 
    for idx, n_exp in enumerate(expert_counts):
        print(f"\n  Training N={n_exp} experts...")
        model = ButterflyMoE_LM(vocab_size, d_model, num_layers, n_exp, min(2, n_exp)).to(device)
        _, best_ppl = train_model(model, train_loader, val_loader, device,
                                   num_epochs=num_epochs, model_name=f'bmoe_{n_exp}exp',max_steps=MAX_STEPS)
 
        # --- accuracy / memory ---
        mem_bmoe = calculate_memory_mb(d_model, d_model * 4, n_exp, 'butterfly')
        mem_std  = calculate_memory_mb(d_model, d_model * 4, n_exp, 'standard')
        compression = mem_std / mem_bmoe
        results.append({
            'n_experts': n_exp,
            'ppl': best_ppl,
            'memory_bmoe_mb': mem_bmoe,
            'memory_std_mb': mem_std,
            'compression': compression,
        })
        print(f"  N={n_exp}: PPL={best_ppl:.2f}, Memory={mem_bmoe:.3f}MB, "
              f"Compression={compression:.1f}x")
 
        # --- expert diversity ---
        model.eval()
        first_moe = model.blocks[0].moe
        w_base_quant = BitNetQuantize.apply(first_moe.w_base)
 
        test_input = torch.randint(0, min(vocab_size, 50257), (1, 32), device=device)
        x_embed = model.embed(test_input)
 
        n_show = min(n_exp, 32)
        expert_outputs = []
        with torch.no_grad():
            for i in range(n_show):
                out = first_moe.experts[i](x_embed, w_base_quant)
                expert_outputs.append(out.flatten())
 
        sim_matrix = torch.zeros(n_show, n_show)
        for i in range(n_show):
            for j in range(n_show):
                sim_matrix[i, j] = F.cosine_similarity(
                    expert_outputs[i].unsqueeze(0), expert_outputs[j].unsqueeze(0)
                ).item()
 
        off_diag = sim_matrix[~torch.eye(n_show, dtype=bool)].mean().item()
        diversity = 1 - off_diag
        diversity_scores.append(diversity)
        print(f"  N={n_exp}: diversity score = {diversity:.4f} (off-diag similarity = {off_diag:.4f})")
 
        ax = axes_div[idx]
        im = ax.imshow(sim_matrix.cpu().numpy(), cmap='RdYlGn_r', vmin=0, vmax=1)
        ax.set_title(f'N={n_exp} experts\ndiversity={diversity:.3f}', fontsize=11)
        ax.set_xlabel('Expert Index'); ax.set_ylabel('Expert Index')
        plt.colorbar(im, ax=ax, fraction=0.046)
 
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
 
    # --- Plot 1: accuracy vs memory ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    mem_bmoe_list = [r['memory_bmoe_mb'] for r in results]
    ppl_list      = [r['ppl']            for r in results]
    comp_list     = [r['compression']    for r in results]
    n_list        = [r['n_experts']      for r in results]
 
    axes[0].plot(mem_bmoe_list, ppl_list, 'o-', color='#2196F3', linewidth=2,
                 markersize=10, label='ButterflyMoE')
    for i, n in enumerate(n_list):
        axes[0].annotate(f'N={n}', (mem_bmoe_list[i], ppl_list[i]),
                         textcoords='offset points', xytext=(5, 5), fontsize=9)
    axes[0].set_xlabel('Memory Footprint (MB)'); axes[0].set_ylabel('Val Perplexity (lower=better)')
    axes[0].set_title('Accuracy vs Memory Tradeoff'); axes[0].grid(alpha=0.3); axes[0].legend()
 
    axes[1].bar([str(n) for n in n_list], comp_list, color='#4CAF50', edgecolor='black')
    axes[1].set_xlabel('Number of Experts'); axes[1].set_ylabel('Compression Ratio vs Standard MoE')
    axes[1].set_title('Compression Ratio Improves with Expert Count')
    axes[1].grid(alpha=0.3, axis='y')
    for i, v in enumerate(comp_list):
        axes[1].text(i, v + 1, f'{v:.1f}x', ha='center', fontweight='bold')
 
    plt.tight_layout()
    plt.savefig('exp2_accuracy_vs_memory.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✓ Saved: exp2_accuracy_vs_memory.png")
