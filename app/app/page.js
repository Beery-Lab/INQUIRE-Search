"use client";
import { useState, useEffect, useRef } from 'react';
import Link from 'next/link'
import * as Icon from 'react-bootstrap-icons';
import { QueryImageGrid } from './query-components';
import { BadgeGreen } from './components';



function QueryNavBar({ onFormSubmit, onDownloadSubmit, images, imageCount=1000,
                       mapSpeciesToCommonName, useNaiveKnn=false, bgColor="slate-800",
                       title="INQUIRE-Search Demo: Use Text to Search Natural World Images",
                       onShowFilters=(() => {}) }) {
  const [loading, setLoading] = useState(false);
  const [value, setValue] = useState('');
  const [showInfo, setShowInfo] = useState(false);

  const exampleQueries = [
    'California condor with green \'26\' on its wing',
    'A close-up of a whale fluke',
    'hermit crab using plastic waste as its shell'
  ]
  async function onSubmit(event) {
    setLoading(true)
    await onFormSubmit(event);
    setLoading(false)
  }

  return (
    <div className={`flex border-b border-b-slate-200 bg-${bgColor} text-white`}>
      <div className="grow px-6 py-5">
        <div className='flex items-center'>
            <div className='grow'>
                <div className='flex items-center mb-2'>
                <h1 className="grow text-2xl">
                    {title}
                </h1>
                
                </div>
                <p className="text-sm mb-2">
                Enter the concepts you're interested in to find relevant iNaturalist photos with CLIP zero-shot retrieval. <br />
	  	</p>
	  {/*
                You can also use the advanced filters to filter by any of the 10,000 species in iNat24. <br />
                </p>
                {/* <p className='text-sm mt-1'>
                    <i>DISCLAIMER: Retrieval uses an open-source model trained on web data. This model may contain harmful biases.</i>
                </p> */}
            </div>
            </div>
            <form onSubmit={onSubmit}>
              <div className='mt-2'>
                    <span className='me-2 text-sm'>Examples: </span> 
                    {exampleQueries.map((q, idx) => (
                        <button key={idx} type='button'
                            className='border border-slate-500 text-sm px-3 py-1 mx-1 mt-2 rounded-full bg-slate-700 text-slate-200 hover:bg-slate-600 hover:text-white' 
                            onClick={(e) => {setValue(q); setTimeout(() => {e.target.form.requestSubmit();}, 300); }}
                            >{q}</button>
                    ))}
                </div>
                <div className='flex mt-4 mb-1 h-10 items-center'>
                    <div className='grow flex items-center bg-white rounded-lg text-slate-800 overflow-hidden'>
                    <div className='h-full flex items-center px-1 ms-2'>
                        <Icon.Search className='shrink-0 w-5 h-5'/>
                    </div>
                    <input 
                        type="text"
                        value={value}
                        name="user_input"
                        onChange={e => { setValue(e.currentTarget.value); }}
                        className='grow px-3 py-2 text-md outline-none'
                        placeholder='Some search query...'
                        maxLength={256}
                    />
                    <input 
                        type="hidden"
                        name="image_count"
                        value={imageCount} />
                    </div>
                    <button disabled={value == ''} type="submit" className='h-full shrink-0 px-2 ms-2 flex items-center border border-slate-200 rounded-lg disabled:text-slate-400 disabled:bg-slate-500 hover:bg-slate-600'>
                    <Icon.ArrowRightShort className='w-8 h-8' />{/* Search */}
                    </button>
                    {loading && (
                    <div className='ms-4 flex items-center'>
                        <Icon.ArrowRepeat
                        className='w-8 h-8 animate-spin' />
                        <p className='ms-2'>Loading...</p>
                    </div>
                    )}
                    {/* Advanced Filters — visible when enabled */}
                    <button type="button" className='shrink-0 flex items-center rounded-lg hover:bg-slate-600 text-sm px-2 ms-3 h-8 '
                            onClick={() => {onShowFilters()}}>
                      <Icon.Filter className='w-5  h-5  inline me-1'/> Advanced Filters
                    </button>
                </div>
                <div className='hidden'>
                    <div className='shrink-0 flex items-end'>
                        <div className='mt-2 inline-flex justify-end items-center px-2 py-1 rounded bg-white bg-opacity-20'>
                        <input type="checkbox" name='use_naive_knn' id='use_naive_knn' defaultChecked={useNaiveKnn} className='w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 cursor-pointer'/>
                        <label htmlFor="use_naive_knn" className='ms-2 text-sm cursor-pointer'>Use perfect search. This is very slow.</label>
                        </div>
                    </div>
                </div>
                
            </form>
        </div>
        <div className="p-5 shrink-0 w-72">
          <div className='h-full flex flex-col items-center justify-center border-s border-slate-500'>
            <p className={'text-xs px-4 mb-2 ' + (images.length <= 0 ? 'text-slate-400' : 'text-slate-200')}>
              Download your search results, image metadata, and marked images as a CSV file.
            </p>
            <button disabled={images.length <= 0} className='px-4 py-2 my-2 text-sm bg-white text-slate-800 rounded-lg hover:bg-slate-200 disabled:bg-slate-400 disabled:text-slate-600' onClick={onDownloadSubmit}>
              <Icon.Download className='w-5 h-5 me-2 inline'/>
              Export Search Results
            </button>
          </div>
      </div>
    </div>
  )
}


function AutocompleteForm({ suggestions, value: externalValue, onChange }) {
  const [value, setValue] = useState("");
  const [currentSuggestions, setCurrentSuggestions] = useState([]);
  const [activeIdx, setActiveIdx] = useState(-1);
  const [selectedSpecies, setSelectedSpecies] = useState("");
  const dropdown = useRef(null);

  // Update internal value when external value changes
  useEffect(() => {
    if (externalValue !== undefined && externalValue !== selectedSpecies) {
      setValue(externalValue);
      setSelectedSpecies(externalValue);
    }
  }, [externalValue]);

  useEffect(() => {
    if (onChange) {
      onChange(selectedSpecies);
    }
  }, [selectedSpecies]);

  function getSuggestionLongName(suggestion) {
    return (
      suggestion['common_name'] 
      ? `${suggestion['common_name']} (${suggestion['name']})` 
      : suggestion['display'] || suggestion['name']
    )
  }

  function onKeyDown(e) {
    if (e.key === "Enter") {
      e.preventDefault();
      if (activeIdx >= 0) {
        const suggestion = currentSuggestions[activeIdx];
        setValue(suggestion['name']);
        setSelectedSpecies(suggestion['name']);
        setCurrentSuggestions([]);
        setActiveIdx(-1);
      }
    } else if (e.key === "ArrowDown") {
      setActiveIdx(Math.min(activeIdx + 1, currentSuggestions.length - 1));
    } else if (e.key === "ArrowUp") {
      setActiveIdx(Math.max(activeIdx - 1, 0));
    }
  }

  function fetchSuggestions(query) {
    fetch(`/api/autocomplete?query=${encodeURIComponent(query)}`)
      .then((resp) => resp.text())
      .then((dataString) => {
        return JSON.parse(dataString.replace(/\bNaN\b/g, "null"));
      })
      .then((data) => {
        setActiveIdx(data.length ? 0 : -1);
        setCurrentSuggestions(data);
      })
      .catch((err) => console.error(err));
  }

  function showSuggestions() {
    if (selectedSpecies || !value.trim()) {
      setCurrentSuggestions([]);
      setActiveIdx(-1);
      return;
    }
    fetchSuggestions(value);
  }

  function onSuggestionClick(suggestion) {
    setValue(suggestion['name']);
    setSelectedSpecies(suggestion['name']);
    setCurrentSuggestions([]);
  }

  function onInputValueChange(e) {
    const newVal = e.currentTarget.value;
    setValue(newVal);
    if (newVal !== selectedSpecies) setSelectedSpecies("");
  }

  useEffect(() => {
    showSuggestions();
  }, [value]);

  useEffect(() => {
    if (!currentSuggestions.length) return;
    function handleClick(event) {
      if (dropdown.current && !dropdown.current.contains(event.target)) {
        setCurrentSuggestions([]);
      }
    }
    window.addEventListener("click", handleClick);
    return () => window.removeEventListener("click", handleClick);
  }, [currentSuggestions]);

  return (
    <div className='grow flex items-center bg-white rounded-lg text-slate-800 px-1'>
      
      <div className='h-full flex items-center ms-1 text-lg'>
        {selectedSpecies ? <span className='text-green-600'>✓</span> : <span className='text-lg'>🔎</span>}
      </div>
      <input
        type="text"
        value={value}
        name="species_full_name"
        autoComplete="off"
        onFocus={showSuggestions}
        onChange={onInputValueChange}
        onKeyDown={onKeyDown}
        className={'grow px-2 py-1.5 me-2 text-xs outline-none font-normal text-slate-800'}
        placeholder='Canis lupus'
      />
      {/* species hidden input removed */}
      {currentSuggestions.length > 0 && (
        <div className='h-0'>
        <div
          ref={dropdown}
          className='w-[480px] absolute top-0 mt-2 ms-2 left-72 z-10 rounded-lg overflow-hidden bg-white border border-slate-300 shadow-lg'
        >
          <div className='bg-slate-200 flex items-center'>
            <div className='grow text-sm font-bold px-4 py-2'>
              Suggestions
            </div>
            <div className='shrink-0 text-sm px-4 py-2'>
              <span className='text-xs font-medium'>Use ↑↓ to navigate and [enter] to select</span>
            </div>
          </div>
          {currentSuggestions.map((suggestion, idx) => (
            <div
              key={idx}
              onClick={() => onSuggestionClick(suggestion)}
              className={
                `px-4 py-2 cursor-pointer text-sm border-b border-b-slate-300 last:border-b-0 ` +
                (idx === activeIdx ? 'bg-blue-500 text-white' : 'hover:bg-slate-100')
              }
            >
              {getSuggestionLongName(suggestion)}
              <span className='text-xs opacity-80 text-white font-medium ms-2 bg-slate-700 bg-opacity-80 px-1 py-0.5 rounded-lg'>
                {suggestion['rank']}
              </span>
            </div>
          ))}
        </div>
        </div>
      )}
      
    </div>
  );
}

function DetailView({ image, imageIdx, numImagesShown, mapSpeciesToCommonName }) {
  return (
    <div className='w-full h-full text-center flex flex-col'>
      <div className='grow min-h-0'>
        <img src={image.src} className='object-contain w-full h-full p-4'/>
      </div>
      <div className='shrink-0 pb-1 text-slate-600 text-sm'>
        Image {imageIdx+1} of {numImagesShown}
      </div>
      
    </div>
  );
}

function ResetButton({ onClick }) {
  return (
    <button
      className='text-xs px-2 py-1 underline rounded hover:bg-slate-300 flex items-center text-slate-700 font-medium'
      onClick={onClick}
      type='button'
    >
      Reset
    </button>
  );
}

function FilterSection({ children, title, onReset=null }) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className='mb-4 pb-4 border-b border-slate-400'>
      <div className='flex items-center justify-between mb-1'>
        <div
          className='text-md text-slate-800 font-bold cursor-pointer'
          onClick={() => setCollapsed(!collapsed)}
        >
          <Icon.ChevronDown className={`inline w-4 h-4 me-2 ${collapsed ? '-rotate-90' : ''}`}/>
          {title}
        </div>
        {onReset && <ResetButton onClick={onReset} />}
      </div>
      {!collapsed && children}
    </div>
  );
}

function FilterSideBar({ mapSpeciesToCommonName, onFilterChange }) {
  const [strengthValue, setStrengthValue] = useState(1);

  const [months, setMonths] = useState(Array(12).fill(false));
  const [monthFilter, setMonthFilter] = useState([]);
  
  // Species filter state
  const [species, setSpecies] = useState("");
  const [speciesFilter, setSpeciesFilter] = useState(null);
  
  // Location filter state
  const [latMin, setLatMin] = useState("");
  const [latMax, setLatMax] = useState("");
  const [lonMin, setLonMin] = useState("");
  const [lonMax, setLonMax] = useState("");
  const [locationFilter, setLocationFilter] = useState(null);

  const monthToString = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
  ];

  // Build species long names list for autocomplete
  const speciesLongNames = Object.entries(mapSpeciesToCommonName || {}).map(([scientificName, commonName]) => ({
    name: scientificName,
    display: `${commonName} (${scientificName})`
  }));

  function toggleMonth(idx) {
    console.log('toggled month: ' + idx);
    const updated = [...months];
    updated[idx] = !updated[idx];
    console.log(updated);
    setMonths(updated);
  }

  useEffect(() => {
    // Active the month filter only if neither all months are selected nor none are selected
    const activeMonths = months.map((active, i) => (active ? i + 1 : null)).filter((i) => i);
    console.log('active months: ', activeMonths);
    setMonthFilter(activeMonths.length < 12 && activeMonths.length > 0 ? {'months': activeMonths} : null);
    console.log(activeMonths);
  }, [months]);
  
  useEffect(() => {
    // Update species filter when species changes
    if (species && species.trim()) {
      setSpeciesFilter({ species: species });
    } else {
      setSpeciesFilter(null);
    }
  }, [species]);
  
  useEffect(() => {
    // Update location filter when lat/lon values change
    const lat1 = parseFloat(latMin);
    const lat2 = parseFloat(latMax);
    const lon1 = parseFloat(lonMin);
    const lon2 = parseFloat(lonMax);
    
    if (!isNaN(lat1) && !isNaN(lat2) && !isNaN(lon1) && !isNaN(lon2)) {
      setLocationFilter({
        use_geo_filters: true,
        latitudeMin: lat1,
        latitudeMax: lat2,
        longitudeMin: lon1,
        longitudeMax: lon2
      });
    } else {
      setLocationFilter(null);
    }
  }, [latMin, latMax, lonMin, lonMax]);


  useEffect(() => {
    // only add filters that are not empty
    const filters = {
      ...(monthFilter ? monthFilter : {}),
      ...(speciesFilter ? speciesFilter : {}),
      ...(locationFilter ? locationFilter : {}),
      'strength': strengthValue,
    };
    console.log("New filters: ", filters);
    onFilterChange(filters);
    console.log(filters);
  }, [monthFilter, speciesFilter, locationFilter, strengthValue]);

  return (
    <div className='w-72 shrink-0 bg-slate-100 border-r border-slate-300 px-5 py-6 overflow-y-scroll'>
      <h2 className='text-xl font-medium mb-4'>Advanced Filters</h2>
      
      {/* Species filter */}
      <FilterSection 
        title='Species'
        onReset={() => setSpecies("")}
      >
        <div className='text-xs mt-1 mb-2'>Search for a species by its common name or scientific name.</div>
        <AutocompleteForm 
          suggestions={speciesLongNames} 
          value={species}
          onChange={(val) => setSpecies(val)} 
        />
        <div className='text-xs mt-2 mx-1'>
          {speciesFilter ? (
            <BadgeGreen className='text-xs'>Filter is active</BadgeGreen>
          ) : (
            <p className='text-slate-600 italic'>Filter is currently not set.</p>
          )}
        </div>
      </FilterSection>

      {/* Location filter */}
      <FilterSection 
        title='Location (Latitude and Longitude)'
        onReset={() => {setLatMin(""); setLatMax(""); setLonMin(""); setLonMax("");}}
      >
        <div className='text-xs mt-1 mb-2'>Filter by geographic bounding box.</div>
        <div className='grid grid-cols-2 gap-2 text-xs'>
          <div>
            <label className='block mb-1'>Min Lat</label>
            <input 
              type="number" 
              step="0.01"
              value={latMin}
              onChange={(e) => setLatMin(e.target.value)}
              className='w-full px-2 py-1 border border-slate-300 rounded'
              placeholder="-90"
            />
          </div>
          <div>
            <label className='block mb-1'>Max Lat</label>
            <input 
              type="number" 
              step="0.01"
              value={latMax}
              onChange={(e) => setLatMax(e.target.value)}
              className='w-full px-2 py-1 border border-slate-300 rounded'
              placeholder="90"
            />
          </div>
          <div>
            <label className='block mb-1'>Min Lon</label>
            <input 
              type="number" 
              step="0.01"
              value={lonMin}
              onChange={(e) => setLonMin(e.target.value)}
              className='w-full px-2 py-1 border border-slate-300 rounded'
              placeholder="-180"
            />
          </div>
          <div>
            <label className='block mb-1'>Max Lon</label>
            <input 
              type="number" 
              step="0.01"
              value={lonMax}
              onChange={(e) => setLonMax(e.target.value)}
              className='w-full px-2 py-1 border border-slate-300 rounded'
              placeholder="180"
            />
          </div>
        </div>
        <div className='text-xs mt-2 mx-1'>
          {locationFilter ? (
            <BadgeGreen className='text-xs'>Filter is active</BadgeGreen>
          ) : (
            <p className='text-slate-600 italic'>Filter is currently not set.</p>
          )}
        </div>
      </FilterSection>


      <FilterSection 
        title='Month'
        onReset={() => {setMonths(Array(12).fill(false));}}
        >
        <div className='text-xs mt-1 mb-2'>Choose which months to include in the search.</div>
        
        <div className="flex flex-wrap text-xs gap-1 font-bold mb-2">
          {months.map((active, idx) => (
            <button
              key={idx}
              type="button"
              className={
                active
                  ? "px-2 py-1 bg-slate-800 text-white rounded border"
                  : "px-2 py-1 bg-slate-200 text-slate-400 rounded border border-slate-300"
              }
              onClick={() => toggleMonth(idx)}
            >
              {monthToString[idx]}
            </button>
          ))}
        </div>
        <div className='text-xs mt-1 mx-1'>
          {monthFilter ? (
            <BadgeGreen className='text-xs mt-1'>Filter is active</BadgeGreen>
          ) : (
            <p className='text-slate-600 italic'>Filter is currently not set or not valid.</p>
          )}
        </div>
      </FilterSection>

    </div>
  );
}

export default function Home({ title, useNaiveKnn=false, bgColor='slate-800', imageSearchCount=200 }) {
  const DEFAULT_IMAGES_PER_PAGE = 50;
  const [selectedIds, setSelectedIds] = useState([]);
  const [showDetailView, setShowDetailView] = useState(true);
  const [images, setImages] = useState([]);
  const [hoveredImageIdx, setHoveredImageIdx] = useState(-1);
  const [sliderValue, setSliderValue] = useState(1);
  const [showSubmitModal, setShowSubmitModal] = useState(false);
  const [numImagesShown, setNumImagesShown] = useState(DEFAULT_IMAGES_PER_PAGE)
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [advancedFilters, setAdvancedFilters] = useState({});

  const [mapSpeciesToCommonName, setMapSpeciesToCommonName] = useState({})

  const gridSizeRange = [6, 10, 14, 18, 22, 30];

  async function onFormSubmit(event) {
    event.preventDefault()
 
    const queryValue = event.currentTarget.elements.namedItem('user_input').value
  // species filter removed; no species value submitted
  console.log('submitted with value: ' + queryValue);

    const formData = new FormData(event.currentTarget);
    // for (const key in advancedFilters) {
    //   formData.append(key, advancedFilters[key]);
    // }
    console.log('advanced filters: ', advancedFilters);
    formData.append('filters', JSON.stringify(advancedFilters));

    try {
      console.log('=== Starting fetch ===');
      const response = await fetch('/process_query', {
        method: 'POST',
        body: formData,
      })
      console.log('=== Fetch completed, response status:', response.status, '===');

      if (!response.ok) {
        console.error(`HTTP Error! status: ${response.status}`)
        const errorText = await response.text();
        console.error('Error response:', errorText);
        return
      }
   
      console.log('=== Parsing JSON response ===');
      const data = await response.json();
      console.log('=== Response data ===', data);
      console.log('=== img_ids type:', typeof data['img_ids'], 'length:', data['img_ids']?.length);
      
      if (!data['img_ids']) {
        console.error('=== ERROR: img_ids is missing from response! ===');
        console.error('Available keys:', Object.keys(data));
        throw new Error('Response missing img_ids field');
      }
      
      console.log('=== Mapping images ===');
      const newImages = data['img_ids'].map((imid, i) => ({
        'id': imid,
        // 'photo_id': data['photo_ids'][i],
        'name': data['img_names'][i],
        // 'species': data['img_names'][i] ? data['img_names'][i].split("/")[1].split("_").slice(-2).join(" ") : '---',
        'species': data['species'][i],
        'src': data['img_urls'][i],
        'score': data['retrieval_scores'][i],
        'latitude': data['latitudes'][i],
        'longitude': data['longitudes'][i],
        'month': data['months'][i],
      }))
    console.log('=== Images mapped successfully, count:', newImages.length, '===');
    
    console.log('=== Getting species names ===');
    const species_names = newImages.map((im) => (mapSpeciesToCommonName[im.species] ?? im.species));
    console.log('=== Species names retrieved ===');
    
    console.log('=== Image URLs (first 5) ===');
    console.log(newImages.slice(0, 5).map((im) => im['src']).join('\n'));
    console.log('=== Full first image object ===');
    console.log(newImages[0]);
    console.log('=== Total images received ===', newImages.length);
    console.log(species_names.slice(0, 50).join('\n'));
    console.log(mapSpeciesToCommonName)
    setImages(newImages);
    setHoveredImageIdx(-1);
    setNumImagesShown(DEFAULT_IMAGES_PER_PAGE)

    console.log(newImages)
    console.log(data)

    // Finally, submit the data for the query value to add it to the list of submitted queries
    // Use formData API to be backwards-compatible with flask version of frontend
    const queryFormData  = new FormData();
    queryFormData.append('data', JSON.stringify({
  query: queryValue,
      time: new Date().getTime(),
      filters: advancedFilters,
      selectedIds: selectedIds,
    }))
    fetch('/submit_data', {
      method: 'POST',
      body: queryFormData,
    })

    setSelectedIds([]);
    } catch (error) {
      console.error('=== Error in onFormSubmit ===', error);
      console.error('Error stack:', error.stack);
      alert('An error occurred while processing the search. Check the console for details.');
    }
  }

  // Alias for backward compatibility with QueryNavBar component
  const onSubmit = onFormSubmit;

  function downloadCSV() {
    if (!images.length) return;

    const rows = images.map(image => ({
      photo_id: image.id,
      species: image.species || '',
      marked: selectedIds.includes(image.id) ? 1 : 0,
      image_url: `https://inquire-search.s3.us-east-2.amazonaws.com/rerank_arxiv/${image.id}`,
      latitude: image.latitude || '',
      longitude: image.longitude || '',
      month: image.month || '',
    }));

    const csvContent = [
      ['photo_id', 'species', 'marked', 'image_url', 'latitude', 'longitude', 'month'].join(','),
      ...rows.map(row => [row.photo_id, row.species, row.marked, row.image_url, row.latitude, row.longitude, row.month].join(','))
    ].join('\n');

    // create file in browser
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);

    // create "a" element with href to this file
    link.setAttribute('href', url);
    link.setAttribute('download', 'search_results.csv');
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();

    // clean up "a" element & remove ObjectURL
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  function addSelectedId(newId) { setSelectedIds((ids) => [...ids, newId]) }
  function removeSelectedId(newId) { setSelectedIds((ids) => ids.filter(id => id !== newId)) }
  function selectAll() { setSelectedIds(images.slice(0, numImagesShown).map((im) => im.id)) }

  function toggleSelected(id) {
    console.log('toggled: ' + id);
    if (selectedIds.includes(id)) {
      removeSelectedId(id)
    } else {
      addSelectedId(id)
    }
  }


  function onImageHover(id) {
    const imageIdx = images.findIndex((im) => im.id == id);
    setHoveredImageIdx(imageIdx);
  }

  function onShowMoreImages() {
    setNumImagesShown(Math.min(numImagesShown + DEFAULT_IMAGES_PER_PAGE, images.length));
  }

  function onShowFewerImages() {
    setNumImagesShown(Math.max(DEFAULT_IMAGES_PER_PAGE * (Math.ceil(numImagesShown/DEFAULT_IMAGES_PER_PAGE) - 1), DEFAULT_IMAGES_PER_PAGE));
  }
  
  function showAll() {
    setNumImagesShown(images.length);
  }

  function scrollToImage(im, block = "end") {
    if (!im) return;
    const element = document.getElementById(`im-${im.id}`)
    if (!element) return;
    element?.scrollIntoView({ behavior: "smooth", block: block, inline: "nearest" });
  }

  useEffect(() => {
    const keyDownHandler = (e) => {
      const formElements = ['INPUT', 'TEXTAREA', 'SELECT', 'OPTION'];

      // Ignore key presses within form elements, like inputs
      if (formElements.includes(e.target.tagName) || images.length == 0) { //  || !showDetailView
        // Do nothing
      } else if (e.key == "ArrowRight") {
        e.preventDefault();
        // setHoveredImageIdx((idx) => (idx >= 0 ? Math.min(idx+1, numImagesShown-1) : idx))
        const newIdx = hoveredImageIdx >= 0 ? Math.min(hoveredImageIdx+1, numImagesShown-1) : hoveredImageIdx;
        setHoveredImageIdx(newIdx)
        scrollToImage(images[newIdx], "center")
      } else if (e.key == "ArrowLeft") {
        e.preventDefault();
        // setHoveredImageIdx((idx) => (idx >= 0 ? Math.max(idx-1, 0) : idx))
        const newIdx = hoveredImageIdx >= 0 ? Math.max(hoveredImageIdx-1, 0) : hoveredImageIdx;
        setHoveredImageIdx(newIdx)
        scrollToImage(images[newIdx], "center")

      } else if (e.key == "Enter" || e.key == " ") {
        e.preventDefault();
        toggleSelected(images[hoveredImageIdx].id)
      }
    }
    document.addEventListener("keydown", keyDownHandler);

    // clean up
    return () => {
      document.removeEventListener("keydown", keyDownHandler);
    };
  }, [hoveredImageIdx, numImagesShown, selectedIds]);


  const loadSpeciesNames = async () => {
    const response = await fetch(`/map_species_to_common_name.json`);
    if (!response.ok) {
        throw new Error(`HTTP Error! status: ${response.status}`)
    }
    const entries = await response.json();
    setMapSpeciesToCommonName(entries);
    // console.log(Object.values(entries).slice(0, 100))
  }

  // On initial page load, load species common names
  useEffect(() => {
    loadSpeciesNames().catch((e) => {
        console.error('An error occured while fetching species data: ', e)
    })
  }, [])

  return (
    <main className="flex flex-col h-[100vh] w-full min-w-[1000px]">
      {/* <p className="fixed left-0 top-0 flex w-full justify-center border-b border-gray-300 bg-gradient-to-b from-zinc-200 pb-6 pt-8 backdrop-blur-2xl dark:border-neutral-800 dark:bg-zinc-800/30 dark:from-inherit lg:static lg:w-auto  lg:rounded-xl lg:border lg:bg-gray-200 lg:p-4 lg:dark:bg-zinc-800/30">
        Go to the review page <a href='/review' className='mx-1 underline'>here</a>.
      </p> */}
    
      <QueryNavBar
        onFormSubmit={onSubmit}
        onDownloadSubmit={downloadCSV}
        // selectedCount={selectedIds.length}
        images={images}
        mapSpeciesToCommonName={mapSpeciesToCommonName}
        imageCount={imageSearchCount}
        useNaiveKnn={useNaiveKnn}
        bgColor={bgColor}
        title={title}
  // Allow the nav bar button to toggle the advanced filters sidebar
  onShowFilters={() => setShowAdvanced(!showAdvanced)}
      />
      <div className='grow flex flex-row h-full overflow-hidden relative'>

        {showAdvanced && (
          <FilterSideBar 
            mapSpeciesToCommonName={mapSpeciesToCommonName}
            onFilterChange={(filters) => {setAdvancedFilters(filters)}}
            />
        )}
        

        <div className='grow flex flex-col'>
          {images.length > 0 && (
            <div className='flex pt-2 pb-2 items-center bg-slate-100 border-b border-slate-200'>
              <div className='grow flex items-center'>
                <h2 className='text-sm ms-6 font-normal'>
                  Retrieved {numImagesShown} Images
                  <span className='ms-4 px-4 border-s border-slate-400 font-medium'>
                    Click on an image to mark it. &nbsp;
                    <u>Marked <b>{selectedIds.length} of {numImagesShown}</b> images.</u>
                  </span>
                  
                </h2>
                <button className='shrink-0 flex items-center px-3 h-7 ms-2 text-xs border border-slate-600 text-slate-800 rounded-lg hover:bg-slate-200'
                  onClick={() => {selectAll()}}>
                  <Icon.CheckAll className='w-5 h-5 me-2'/>
                  Select All
                </button>
                <button className='shrink-0 flex items-center px-3 h-7 ms-2 text-xs border border-slate-600 text-slate-800 rounded-lg hover:bg-slate-200'
                  onClick={() => {showAll()}}>
                  <Icon.ArrowsExpand className='w-4 h-4 me-2'/>
                  Show All
                </button>
              </div>
              
              
              <div className='shrink-0 flex px-3 py-1 items-center rounded-lg'>
                <Icon.GridFill className='w-4 h-4 me-1'/>
                <p className='text-sm font-medium me-3'>
                  Grid Size
                </p>
                <input id="default-range" type="range" value={sliderValue} min={0} max={gridSizeRange.length-1} 
                  onChange={e => { setSliderValue(e.currentTarget.value); }}
                  className="w-32 h-2 bg-slate-300 rounded-lg appearance-none cursor-pointer dark:bg-slate-700" />
              </div>

              <button className='shrink-0 flex items-center px-3 py-1 mx-4 text-xs border border-slate-600 text-slate-800 rounded-lg hover:bg-slate-200'
                onClick={() => {setShowDetailView(!showDetailView)}}>
                <Icon.Image className='w-4 h-4 me-2'/>
                Show/Hide Detail View
              </button>
            </div>
          )}
          {images.length > 0 ? (
            <div className='grow flex overflow-hidden'>
              <div className='grow basis-0 overflow-scroll h-full'>
                <QueryImageGrid
                  images={images.slice(0, numImagesShown)}
                  selectedIds={selectedIds}
                  gridSize={gridSizeRange[sliderValue]}
                  hoveredImageIdx={hoveredImageIdx}
                  // onImageClick={() => {}}
                  onImageClick={toggleSelected} 
                  onImageHover={onImageHover}
                />

                <div className='py-5 text-center bg-slate-100 border-t border-t-slate-200'>
                  <button disabled={numImagesShown <= DEFAULT_IMAGES_PER_PAGE} onClick={onShowFewerImages} className='px-4 py-1 mx-2 border border-slate-600 text-slate-800 rounded-lg hover:bg-slate-200 disabled:text-slate-400 disabled:bg-slate-200 disabled:border-slate-300'>
                    <Icon.ChevronContract className='w-4 h-4 me-2 inline'/>
                    Show Less
                  </button>
                  <button disabled={numImagesShown >= images.length} onClick={onShowMoreImages} className='px-4 py-1 mx-2 border border-slate-600 text-slate-800 rounded-lg hover:bg-slate-200 disabled:text-slate-400 disabled:bg-slate-200 disabled:border-slate-300'>
                    <Icon.Grid3x2GapFill className='w-4 h-4 me-2 inline'/>
                    Show More
                  </button>
                </div>
              </div>
              {showDetailView && (
                <div className='grow basis-0 bg-slate-100 border-s border-slate-200'>
                    {hoveredImageIdx != null && hoveredImageIdx >= 0 && (
                      <DetailView image={images[hoveredImageIdx]} imageIdx={hoveredImageIdx} numImagesShown={numImagesShown} mapSpeciesToCommonName={mapSpeciesToCommonName}/>
                    )}
                </div>
              )}
            </div>
            ) : (
              <div className='grow flex items-center justify-center'>
                <p className='text-slate-500'>Waiting for your input...</p>
              </div>
            )
          }
        </div>
      </div>
    </main>
  )
}
