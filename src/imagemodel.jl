abstract type AbstractImagingModel end

struct ImagingModel{I, M<:MeanModels, C, P, PM} <: AbstractImagingModel
    img::I
    mean_model::M
    order::Int
    cache::C
    img_prior::P
    mean_prior::PM
    ftot::Float64
end

struct PolarizedImagingModel{I, M<:MeanModels, C, P, PM} <: AbstractImagingModel
    img::I
    mean_model::M
    order::Int
    cache::C
    img_prior::P
    mean_prior::PM
    ftot::Float64
end


function markov_imaging_model(
    base_dist::Type{<:MarkovRandomField},
    mean_model::ModelAndPrior{<:MeanModel},
    dvis::EHTObservation,
    fovx::Real,
    fovy::Real,
    nx::Int,
    ny::Int,
    ftot::Real;
    x0 = μas2rad(0.0),
    y0 = μas2rad(0.0),
    order = 1,
    pulse = BSplinePulse{3}(),
    )
    g = imagepixels(fovx, fovy, nx, ny, x0, y0)
    cache = create_cache(NFFTAlg(dvis), g, pulse)
    return ImagingModel(
            image_model.model, imagemodel.prior,
            order, cache,
            mean_model.model, mean_model.prior,
            ftot)
end

function matern_imaging_model(
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
    bkgd_prior = truncated(Normal(0.0, 0.5); lower=0.0),
    img_prior = (ρ = truncated(InverseGamma(2.0, -log(0.1)*beam(dvis)/(fovx/nx), lower=1.0, upper=max(nx, ny)),
                 σ = truncated(Normal(0.0, 0.5); lower=0.0, upper = 5.0)),
                 ν = Uniform(0.5, 6.0)
                )
    )


function apply_fluctuations(::MarkovRandomField, ftot, m0::AbstractMatrix, p)
    (;σ, params) = p
    delta = to_simplex(CenteredLR(), σ.*params)
    rast = m0.*delta
    img = (ftot).*rast./sum(rast)
    return img
end

function apply_fluctuations(mat::StationaryMatern, ftot, m0::AbstractMatrix, p)
    (;σ, ρ, ν, params) = p
    delta = to_simplex(CenteredLR(), σ.*mat(ρ, ν, params))
    rast = m0.*delta
    img = (ftot).*rast./sum(rast)
    return img
end


function apply_polarized_fluctuations(::MarkovRandomField, p)
    (;p0, pσ, params) = p
    return logistic.(p0 .+ pσ.*params)
end

function apply_polarized_fluctuations(img::StationaryMatern, p)
    (;p0, pσ, pρ, pν, params) = p
    return logistic.(p0 .+ pσ.*img(pρ, pν, params))
end


function (model::ImagingModel)(x)
    (;img_params, mean_params) = x
    m0 = model.mean_model(mean_params)
    img = apply_fluctuations(mode.image, model.ftot, m0, img_params)
    cimg = ContinuousImage(img, model.cache)
    return cimg
end

function (model::PolarizedImagingModel)(x)
    (;img_params, mean_params, p_params, angparams) = x
    m0 = model.mean_model(mean_params)
    fluc = model.image
    img = apply_fluctuations(fluc, model.ftot, m0, img_params)
    pim = apply_polarized_fluctuations(fluc, p_params)
    m = PoincareSphere2Map(img, pim, angparams, model.cache)
    return m
end
