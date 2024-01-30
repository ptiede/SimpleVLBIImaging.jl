struct ImagingModel{P, C, M<:MeanModels, PM}
    image_prior::P
    order::Int
    cache::C
    mean_model::M
    mean_model_prior::PM
    ftot::Float64
end

function StandardImagingModel(
    dvis::EHTObservation,
    fovx::Real,
    fovy::Real,
    nx::Int,
    ny::Int,
    ftot::Real;
    x0 = μas2rad(0.0),
    y0 = μas2rad(0.0),
    order = 1,
    pulse = BSplinePulse{1}(),
    gauss_fwhm = μas2rad(60.0),
    bkgd_prior = truncated(Normal(0.0, 0.5); lower=0.0)
    )
    g = imagepixels(fovx, fovy, nx, ny, x0, y0)
    cache = create_cache(NFFTAlg(dvis), g, pulse)
    mean_model = SimpleMean(gauss_fwhm, g)
    return ImagingModel(GMRF, order, cache, mean_model, bkgd_prior, ftot)
end


function (model::ImagingModel{<:MarkovRandomField})(x)
    (;c, σ, p, p0, pσ, angparams) = x
    m0 = model.mean_model(x)
    cp = m0 .+ σ.*c.params
    rast = (model.ftot)*(to_simplex(CenteredLR(), cp))
    pim = logistic.(p0 .+ pσ.*p.params)
    m = PoincareSphere2Map(rast, pim, angparams, model.cache)
    x0, y0 = centroid(stokes(m, :I))
    return shifted(m, -x0, -y0)
end

function instrument(θ, metadata)
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
