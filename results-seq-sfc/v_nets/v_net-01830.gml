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
  id 1830
  arrival_time 40464.13251133305
  lifetime 103.80453592740558
  num_nodes 5
  type "path"
  node [
    id 0
    label "0"
    cpu 36
    gpu 37
    rom 1
  ]
  node [
    id 1
    label "1"
    cpu 14
    gpu 26
    rom 29
  ]
  node [
    id 2
    label "2"
    cpu 43
    gpu 11
    rom 12
  ]
  node [
    id 3
    label "3"
    cpu 6
    gpu 32
    rom 2
  ]
  node [
    id 4
    label "4"
    cpu 46
    gpu 16
    rom 9
  ]
  edge [
    source 0
    target 1
    bw 40
  ]
  edge [
    source 1
    target 2
    bw 16
  ]
  edge [
    source 2
    target 3
    bw 33
  ]
  edge [
    source 3
    target 4
    bw 27
  ]
]
