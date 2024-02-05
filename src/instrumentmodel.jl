Base.@kwdef struct InstrumentModel
    log_amplitude_priors = Normal(0.0, 0.1)
    log_amplitude_seg = ScanSeg()
    amplitude_ratio_priors = Normal(0.0, 0.1)
    amplitude_ratio_seg = TrackSeg()
    phase_priors = DiagVonMises(0.0, inv(π^2))
    phase_seg = ScanSeg()
    phase_ratio_priors = DiagVonMises(0.0, inv(π^2))
    d_term_priors = Normal(0.0, 0.1)
    d_term_seg = TrackSeg()
    reference_station = :ALMA
    add_fr = true
end

function stokesi_instrument(θ, metadata)
    (;lg, gp) = θ
    (;ampcache, phasecache) = metadata
    ## Now form our instrument model
    jg  = jonesStokes(exp.(lg),  ampcache)
    jp  = jonesStokes(exp.(1im.*gp),  phasecache)
    return JonesModel(jg*jp)
end



function polarized_instrument(θ, metadata)
    (; lgR, lgr, gpR, gpr, dRx, dRy, dLx, dLy) = θ
    (; tcache, ampcache1, ampcache2, phasecache1, phasecache2, trackcache) = metadata
    R = jonesR(tcache)

    ## Gain product parameters
    gR = exp.(lgR .+ 0.0im)
    gr = exp.(lgr .+ 0.0im)
    Ga1 = jonesG(gR, gR, ampcache1)
    Ga2 = jonesG(fill(complex(1.0), length(gr)), gr, ampcache2)

    egpR = exp.(1im*gpR)
    egpr = exp.(1im*(gpr))
    Gp1 = jonesG(egpR, egpR, phasecache1)
    Gp2 = jonesG(fill(complex(1.0), length(egpr)), egpr, phasecache2)

    D = jonesD(complex.(dRx, dRy), complex.(dLx, dLy), trackcache)
    J =  Ga1*Ga2*Gp1*Gp2*D*R
    return JonesModel(J, tcache)
end

function build_instrument(
    dvis::EHTObservation,
    ::PolarizedImagingModel,
    intmodel::InstrumentModel
    )

    tcache      = ResponseCache(dvis; add_fr=intmodel.add_fr)
    ampcache1   = jonescache(dvis, intmodel.log_amplitude_segment)
    ampcache2   = jonescache(dvis, intmodel.amplitude_ratio_segment)
    phasecache1 = jonescache(dvis, intmodel.phase_segment; autoref=SEFDReference(complex(1.0)))
    phasecache2 = jonescache(dvis, intmodel.phase_ratio_segment;
                                autoref = SingleReference(intmodel.reference_station, complex(1.0))
                            )
    trackcache  = jonescache(dvis, intmodel.dterm_segment)

    instrumentmeta = (;ampcache1, ampcache2,
                       phasecache1, phasecache2,
                       trackcache,
                       tcache
                    )

    dist_amp     = station_tuple(dvis, intmodel.log_amplitude_prior)
    dist_pha     = station_tuple(dvis, intmodel.phase_priors)
    dist_amp_rel = station_tuple(dvis, intmodel.phase_offset_prior)
    dist_dte     = station_tuple(dvis, intmodel.dterm_prior)

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

    return instrumentmeta, instrument_prior, polarized_instrument
end


function build_instrument(
    dvis::EHTObservation,
    ::ImagingModel;
    intmodel::InstrumentModel
    )

    ampcache   = jonescache(dvis, intmodel.log_amplitude_segment)
    phasecache = jonescache(dvis, intmodel.phase_segment; autoref=SEFDReference(complex(1.0)))

    instrumentmeta = (;
                       ampcache,
                       phasecache,
                     )

    dist_amp     = station_tuple(dvis, intmodel.log_amplitude_prior)
    dist_pha     = station_tuple(dvis, intmodel.phase_prior)

    instrument_prior = (
        lg  = CalPrior(dist_amp, ampcache),
        gp  = CalPrior(dist_pha, phasecache),
    )

    return instrumentmeta, instrument_prior, stokesi_instrument
end
