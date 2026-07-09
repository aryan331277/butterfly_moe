def exp4_non_power_of_two(train_loader, val_loader, vocab_size, device,
                           num_layers, top_k, num_epochs):


    # d=512 is power of two (baseline), d=768 and d=1024 are not / are
    configs = [
        {'d_model': 768,  'label': 'd=768 (non-pow2, BERT-base)'},
    ]

    results = []
    for cfg in configs:
        d = cfg['d_model']
        print(f"\n  Testing {cfg['label']}...")
        try:
            model = ButterflyMoE_LM(vocab_size, d, 2, 8, 2).to(device)
            _, best_ppl = train_model(model, train_loader, val_loader, device,
                                       num_epochs=min(num_epochs, 15),
                                       model_name=f'npow2_d{d}',max_steps=MAX_STEPS)
            tput, lat = measure_throughput(model, 16, 64, vocab_size, device, num_iters=50)
            mem = calculate_memory_mb(d, d * 4, 8, 'butterfly')
            results.append({
                'label': cfg['label'], 'd_model': d,
                'ppl': best_ppl, 'throughput': tput,
                'latency_ms': lat, 'memory_mb': mem,
            })
            print(f"  PPL={best_ppl:.2f}, Throughput={tput:.0f} tok/s, Latency={lat:.2f}ms")
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAILED for d={d}: {e}")
            results.append({'label': cfg['label'], 'd_model': d, 'ppl': None,
                            'throughput': None, 'latency_ms': None, 'memory_mb': None})

    # Table
    print("\n" + "-"*75)
    print(f"{'Config':<35} {'PPL':<10} {'Throughput':<15} {'Latency(ms)':<15} {'Mem(MB)':<10}")
    print("-"*75)
    for r in results:
        if r['ppl'] is not None:
            print(f"{r['label']:<35} {r['ppl']:<10.2f} {r['throughput']:<15.0f} "
                  f"{r['latency_ms']:<15.2f} {r['memory_mb']:<10.3f}")
        else:
            print(f"{r['label']:<35} {'FAILED':<10}")

    # Plot
    valid = [r for r in results if r['ppl'] is not None]
    if valid:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        labels = [r['label'].split('(')[0].strip() for r in valid]
        ppls   = [r['ppl'] for r in valid]
        tputs  = [r['throughput'] for r in valid]
        lats   = [r['latency_ms'] for r in valid]

        axes[0].bar(labels, ppls, color='#2196F3', edgecolor='black')
        axes[0].set_ylabel('Validation Perplexity'); axes[0].set_title('PPL across Dimensions')
        axes[0].grid(alpha=0.3, axis='y')

        axes[1].bar(labels, tputs, color='#4CAF50', edgecolor='black')
        axes[1].set_ylabel('Throughput (tok/s)'); axes[1].set_title('Throughput across Dimensions')
        axes[1].grid(alpha=0.3, axis='y')

        axes[2].bar(labels, lats, color='#FF5722', edgecolor='black')
        axes[2].set_ylabel('Latency (ms)'); axes[2].set_title('Latency across Dimensions')
        axes[2].grid(alpha=0.3, axis='y')

        plt.suptitle('Non-Power-of-Two Dimension Robustness', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('exp4_non_power_of_two.png', dpi=150, bbox_inches='tight')
        plt.close()

    return results
