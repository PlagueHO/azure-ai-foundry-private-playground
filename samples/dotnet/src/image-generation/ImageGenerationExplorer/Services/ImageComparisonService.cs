using ImageGenerationExplorer.Models;

namespace ImageGenerationExplorer.Services;

/// <summary>
/// Orchestrates parallel image generation across multiple models.
/// </summary>
public sealed class ImageComparisonService
{
    private readonly IReadOnlyDictionary<ImageApiType, IImageGenerationProvider> _providers;

    /// <summary>
    /// Initializes a new instance of the <see cref="ImageComparisonService"/> class.
    /// </summary>
    public ImageComparisonService(IEnumerable<IImageGenerationProvider> providers)
    {
        ArgumentNullException.ThrowIfNull(providers);
        _providers = providers.ToDictionary(p => p.ApiType);
    }

    /// <summary>
    /// Generates images in parallel for the given prompt across all specified models.
    /// </summary>
    /// <param name="prompt">The text prompt.</param>
    /// <param name="models">Models to generate with, each including its own width/height.</param>
    /// <param name="cancellationToken">A cancellation token.</param>
    /// <returns>A comparison run with results from all models.</returns>
    public async Task<ComparisonRun> CompareAsync(
        string prompt,
        IReadOnlyList<(ImageModelConfig Model, int Width, int Height)> models,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(models);

        var tasks = models.Select(entry =>
        {
            if (!_providers.TryGetValue(entry.Model.ApiType, out var provider))
            {
                return Task.FromResult(ImageGenerationResult.Failure(
                    entry.Model.DeploymentName,
                    entry.Model.DisplayName,
                    $"No provider registered for API type '{entry.Model.ApiType}'.",
                    0));
            }

            return provider.GenerateAsync(prompt, entry.Width, entry.Height, entry.Model, cancellationToken);
        });

        var results = await Task.WhenAll(tasks).ConfigureAwait(false);

        return new ComparisonRun
        {
            Prompt = prompt,
            Results = results,
        };
    }

    /// <summary>
    /// Regenerates a single image for the given prompt and model configuration.
    /// </summary>
    /// <param name="prompt">The text prompt.</param>
    /// <param name="model">The model to generate with.</param>
    /// <param name="width">Output width in pixels.</param>
    /// <param name="height">Output height in pixels.</param>
    /// <param name="cancellationToken">A cancellation token.</param>
    /// <returns>The generation result for the single model.</returns>
    public Task<ImageGenerationResult> GenerateSingleAsync(
        string prompt,
        ImageModelConfig model,
        int width,
        int height,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);

        if (!_providers.TryGetValue(model.ApiType, out var provider))
        {
            return Task.FromResult(ImageGenerationResult.Failure(
                model.DeploymentName,
                model.DisplayName,
                $"No provider registered for API type '{model.ApiType}'.",
                0));
        }

        return provider.GenerateAsync(prompt, width, height, model, cancellationToken);
    }
}
