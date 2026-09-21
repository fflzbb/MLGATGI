import pysam
import numpy as np
import random
from joblib import Parallel, delayed
from tqdm import tqdm


def process_record(record):
    allele_counts = np.zeros(len(record.alleles))
    for sample in record.samples.values():
        genotype = sample['GT']
        for allele in genotype:
            if allele is not None and allele >= 0:
                allele_counts[allele] += 1
    total_alleles = np.sum(allele_counts)
    if total_alleles == 0:
        return None  
    allele_freqs = allele_counts / total_alleles
    maf = np.min(allele_freqs)
    return (record, maf)


def calculate_maf(vcf_file, n_jobs=-1):
    vcf = pysam.VariantFile(vcf_file)
    records = list(vcf)
    mafs = Parallel(n_jobs=n_jobs)(
        delayed(process_record)(record) for record in tqdm(records)
    )
    print(f"MAFs before filtering None: {len(mafs)}")
    mafs = [result for result in mafs if result is not None]
    print(f"MAFs after filtering None: {len(mafs)}")
    return mafs


def group_variants_by_maf(mafs, bins):
    binned_variants = {i: [] for i in range(len(bins) - 1)}
    for record, maf in mafs:
        for i in range(len(bins) - 1):
            if bins[i] <= maf < bins[i + 1]:
                binned_variants[i].append((record, maf))
                break
    return binned_variants


def randomly_replace_with_missing(vcf_file, mafs, output_vcf_file, bins, sample_fraction):
    binned_variants = group_variants_by_maf(mafs, bins)
    inputt = pysam.VariantFile(vcf_file)
    all_variants = []

    # Process each bin
    for bin_index, variants in binned_variants.items():
        sample_size = int(sample_fraction * len(variants))
        selected_variants = random.sample(variants, min(sample_size, len(variants)))
        for record, maf in selected_variants:
            for sample in record.samples:
                if random.random() < sample_fraction:
                    record.samples[sample]['GT'] = (None, None)
                    record.samples[sample].phased = True
            all_variants.append(record)

    all_variants.sort(key=lambda rr: (rr.contig, rr.pos))
    outputt = pysam.VariantFile(output_vcf_file, 'w', header=inputt.header)
    for record in all_variants:
        outputt.write(record)
    inputt.close()
    outputt.close()


bins = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]

for missing_sites_fraction in tqdm([0.7, 0.95, 0.99]):
    vcf_test_file = f'./21k_unique_haps_total_chr19/ceu.model.OutOfAfrica_4J17.gmap.PyrhoCEU_GRCh38.chr.19.50000.test.samples.with.{missing_sites_fraction}.missing.sites.10000.snvs.biallelic.vcf'
    mafs = calculate_maf(vcf_test_file)
    for sample_fraction in tqdm([0.1, 0.2, 0.5, 0.95, 0.99], leave=False):
        output_vcf_file = f'./21k_unique_haps_total_chr19/ceu.model.OutOfAfrica_4J17.gmap.PyrhoCEU_GRCh38.chr.19.50000.test.samples.with.{missing_sites_fraction}.missing.sites.and.{sample_fraction}.random.missingness.10000.snvs.biallelic.vcf'
        randomly_replace_with_missing(vcf_test_file, mafs, output_vcf_file, bins, sample_fraction)
        print(f'Successfully written VCF with missing values to {output_vcf_file}')
