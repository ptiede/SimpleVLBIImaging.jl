struct ImagingProblem{P,D,M,I}
    post::P
    data::D
    model::M
    intmodel::I
end


function ImagingProblem(
    dvis::EHTObservation,
    model::ImagingModel,
    intmodel::InstrumentModel
                        )

    @assert reference_station in stations(dvis) "You reference station $(reference_station) isn't in the array. This isn't valid"

    instrumentmeta, instrument_prior = build_instrument(
                    dvis,
                    model,
                    intmodel
                )

    img_prior = create_image_prior(dvis, model)

    prior = NamedDist(merge(img_prior, instrument_prior))
    lklhd = RadioLikelihood(model, instrument, dvis; instrumentmeta)
    post = Posterior(lklhd, prior)
    return ImagingProblem(post, dvis, model, intmodel)
end
