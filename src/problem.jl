struct ImagingProblem{P,D}
    post::P
    data::D
end


function ImagingProblem(
    dvis::EHTObservation, model::ImagingModel,
    log_amplitude_prior = Normal(0.0, 0.1),
    phase_offset_prior = Normal(0.0, 0.01),
    dterm_prior = Normal(0.0, 0.1),
    log_amplitude_segment = ScanSeg(),
    amplitude_ratio_segment = TrackSeg(),
    phase_segment = ScanSeg(),
    phase_ratio_segment = TrackSeg(),
    dterm_segment = TrackSeg(),
    reference_station = :ALMA
                        )

    @assert reference_station in stations(dvis) "You reference station $(reference_station) isn't in the array. This isn't valid"

    tcache      = ResponseCache(dvis; add_fr=true)
    ampcache1   = jonescache(dvis, log_amplitude_segment)
    ampcache2   = jonescache(dvis, amplitude_ratio_segment)
    phasecache1 = jonescache(dvis, phase_segment; autoref=SEFDReference(complex(1.0)))
    phasecache2 = jonescache(dvis, phase_ratio_segment; autoref = SingleReference(reference_station, complex(1.0)))
    trackcache  = jonescache(dvis, dterm_segment)

    instrumentmeta = (;ampcache1, ampcache2,
                       phasecache1, phasecache2,
                       trackcache,
                       tcache
                    )

    img_prior = create_image_prior(model)

    dist_amp     = station_tuple(dvis, log_amplitude_prior)
    dist_pha     = station_tuple(st, DiagonalVonMises(0.0, inv(π^2)))
    dist_amp_rel = station_tuple(dvis, phase_offset_prior)
    dist_dte     = station_tuple(dvis, dterm_prior)

    instrument_prior = (
        lgR  = CalPrior(dist_amp, ampcache1),
        gpR  = CalPrior(dist_pha, phasecache1),
        lgr  = CalPrior(dist_amp_rel, ampcache2),
        gpr  = CalPrior(dist_pha_rel, phasecache2),
        dRx  = CalPrior(dist_dte, trackcache),
        dRy  = CalPrior(dist_dte, trackcache),
        dLx  = CalPrior(dist_dte, trackcache),
        dLy  = CalPrior(dist_dte, trackcache),
    )

    prior = NamedDist(merge(img_prior, instrument_prior))
    lklhd = RadioLikelihood(model, instrument, dvis; instrumentmeta)
    post = Posterior(lklhd, prior)
    return ImagingProblem(post, dvis)
end
