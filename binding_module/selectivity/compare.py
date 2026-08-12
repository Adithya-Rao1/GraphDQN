def compare_affinities(target_affinity, off_target_affinity):
    kd_ratios = []
    
    if type(target_affinity) == list:
        for target_aff, off_target_aff in zip(target_affinity, off_target_affinity):
            if target_aff == 0: 
                continue
            kd_ratio = off_target_aff / target_aff
            kd_ratios.append(kd_ratio)
        
        return kd_ratios
    
    else:
        kd_ratio = off_target_affinity / target_affinity
        return kd_ratio

if __name__ == "__main__":
    target_affinity = [1.0, 2.0]
    off_target_affinity = [5.0, 3.0]

    result = compare_affinities(target_affinity, off_target_affinity)
    print(result)
