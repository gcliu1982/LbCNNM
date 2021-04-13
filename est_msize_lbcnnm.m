function [m_size] = est_msize_lbcnnm(x, options)
len = length(x);
cds = options.mh_min : options.mh_max; 
cds = cds * options.h; %candidates

%%% stage 1: coarse search
%applying SDR filter
sdr = (len - cds + 1)./cds;
cds = cds(sdr >= options.min_sdr);
if length(cds) < 1
    m_size = min(len + options.h, options.mh_min*options.h);
    return;
end

%forward validation
if length(cds) < 2
    m_size = cds(1);
else
    nb_fold = min(options.nbf, len - cds(2) + 1);
    if nb_fold < 1
       m_size = min(cds);
    else
       m_size = forward_validation_ent(x, options.h, cds, nb_fold, options.gamma, options.nb_bin); 
    end
end

%%% stage 2: fine search
a0 = round(m_size/options.h);
cds = a0 - 1 + options.ss : options.ss : a0 + 1 - options.ss;
cds = round(cds*options.h);
cds = unique(cds(cds >= options.mh_min*options.h));

%applying SDR filter
sdr = (len - cds + 1)./cds;
cds = cds(sdr >= options.min_sdr);
if length(cds) < 2
    m_size = cds(1);
    return;
end

%forward validation
nb_fold = min(options.nbf, len - max(cds) + 1);
if nb_fold >= 1
   m_size = forward_validation_ent(x, options.h, cds, nb_fold, options.gamma, options.nb_bin); 
end

end