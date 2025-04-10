graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 940
  arrival_time 20091.80939885138
  lifetime 3067.4612431624073
  num_nodes 7
  type "path"
  node [
    id 0
    label "0"
    cpu 2
    gpu 27
    rom 38
  ]
  node [
    id 1
    label "1"
    cpu 43
    gpu 35
    rom 20
  ]
  node [
    id 2
    label "2"
    cpu 37
    gpu 13
    rom 21
  ]
  node [
    id 3
    label "3"
    cpu 20
    gpu 6
    rom 45
  ]
  node [
    id 4
    label "4"
    cpu 16
    gpu 27
    rom 31
  ]
  node [
    id 5
    label "5"
    cpu 40
    gpu 2
    rom 3
  ]
  node [
    id 6
    label "6"
    cpu 6
    gpu 29
    rom 38
  ]
  edge [
    source 0
    target 1
    bw 48
  ]
  edge [
    source 1
    target 2
    bw 41
  ]
  edge [
    source 2
    target 3
    bw 11
  ]
  edge [
    source 3
    target 4
    bw 9
  ]
  edge [
    source 4
    target 5
    bw 11
  ]
  edge [
    source 5
    target 6
    bw 26
  ]
]
